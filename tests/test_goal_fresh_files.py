"""CPU file closure. Synthetic fixtures never claim a real B45/model receipt.

Only the three hardcoded historical file SHA values are replaced for a tiny
fixture. An independent negative test uses the actual guard unchanged.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from luban import goal_fresh as fresh
from luban.training_checkpoint import implementation_hashes


class FileClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name).resolve()
        def record(index, purpose, acceptance=False):
            return {'id': 'accept' + str(index) if acceptance else 'new' + str(index),
                'prompt': 'CPU fixture ' + str(index), 'answer': 'OK', 'purpose': purpose,
                'source': 'human', 'approved': True, 'training_allowed': not acceptance,
                'provenance': json.dumps({'draft': False, 'evolution_role': 'new',
                    'generation_check': {'type': 'exact', 'expected': 'OK', 'max_new_tokens': 8},
                    'retention_generation_only': acceptance})}
        material = {'TRAIN': [record(i, 'training') for i in range(24)],
                    'VALID': [record(i, 'validation') for i in range(24, 32)]}
        acceptance = [record(i, 'validation', True) for i in range(8)]
        old = [{'id': 'old' + str(i), 'purpose': 'validation', 'training_allowed': False} for i in range(89)]
        self.mapping = {'new_material': 'new-material.json', 'old_retention_complete': 'old-core-retention.json',
            'old_training_original': 'old-training-original.json', 'old_protection': 'old-protection.json', 'acceptance': 'acceptance.json'}
        self.write('new-material.json', material)
        self.write('acceptance.json', acceptance)
        self.write('old-protection.json', old)
        self.write('old-training-original.json', [])
        self.write('old-core-retention.json', {})
        original = 'c887e507841eb4f0cc81b1dd2fe82f58b4baa80db8a1d30eedb286bc8f36a4d6'
        rows = [{'id': r['id'], 'passed': True} for r in old]
        self.write('historical-receipt.json', {'starting_tensor_sha256': original,
            'training': {'baseline_core_retention_generation': rows}})
        self.write('capture.json', {'policy': 'known-old-generated-states-v1', 'scope': 'known_old_protection_only',
            'original_tensor_sha256': original, 'baseline_tensor_sha256': original,
            'records': old, 'results': rows, 'authorized_record_sha256': {r['id']: fresh.content_sha(r) for r in old},
            'baseline_receipt': self.descriptor('historical-receipt.json')})
        implementation = implementation_hashes({'goal_sequence': {'parent_steps': 0}})
        self.protocol = {**self.mapping, 'manifest': 'manifest.json', 'protocol': fresh.PROTOCOL,
            'status': 'PRE_REGISTERED_NOT_TRAINED', 'segment_steps': 450, 'total_steps': 2700,
            'policy_version': fresh.POLICY, 'new_run_id': 'abcdef123456', 'identity': 'CPU-fixture',
            'source_core': {'version': 'core-b45d965728bb', 'identity': 'CPU-fixture',
                'sha256': 'a2eb54773a4fdc18196acf966c4c21400bbbcd8b811fdc2518b7cbcc40442628'},
            'original_tensor_sha256': original, 'new_counts': {'TRAIN': 24, 'VALID': 8}, 'acceptance_count': 8,
            'old_counts': {'original_training': 83, 'protected_generation': 89},
            'new_implementation_sha256': implementation,
            'new_source_sha256': {'src/luban/' + n: v for n, v in implementation.items()},
            'protection_capture': self.descriptor('capture.json')}
        self.path = self.directory / 'protocol.json'
        self.old_fixture_sha = {field: fresh.common.file_sha(self.directory / self.mapping[field]) for field in fresh.OLD_FILES}
        self.refresh_manifest()
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        (self.directory / name).write_text(json.dumps(data))

    def descriptor(self, name):
        return {'path': str(self.directory / name), 'sha256': fresh.common.file_sha(self.directory / name)}

    def refresh_manifest(self):
        self.write('manifest.json', {'files': {name: {'sha256': fresh.common.file_sha(self.directory / name)}
            for name in self.mapping.values()}})

    def save(self):
        self.path.write_text(json.dumps(self.protocol))

    def load(self):
        with patch.object(fresh, 'OLD_FILES', self.old_fixture_sha):
            return fresh.load_verified(self.path)

    def test_source_closure_and_purpose_vs_rights(self):
        checked = self.load()
        self.assertEqual(len(checked['new_implementation_sha256']), 22)
        self.assertEqual(fresh.metadata(checked)['parent_steps'], 0)

    def test_fake_old_snapshot_rejected_by_unmodified_actual_guard(self):
        with self.assertRaises(ValueError):
            fresh.load_verified(self.path)

    def test_material_byte_tamper_is_rejected(self):
        self.write('new-material.json', [])
        with self.assertRaises(ValueError):
            self.load()

    def test_source_key_omission_not_allowed(self):
        del self.protocol['new_implementation_sha256']['goal_topics.py']
        self.save()
        with self.assertRaises(ValueError):
            self.load()

    def test_budget_not_generic_unbounded(self):
        self.protocol['total_steps'] = 3150
        self.save()
        with self.assertRaises(ValueError):
            self.load()

    def test_strict_duplicate_json_rejected(self):
        self.path.write_text('{"protocol":"bad","protocol":"good"}')
        with self.assertRaises(ValueError):
            self.load()

    def test_symlinked_material_refused(self):
        source = self.directory / 'new-material-original.json'
        source.write_bytes((self.directory / 'new-material.json').read_bytes())
        (self.directory / 'new-material.json').unlink()
        (self.directory / 'new-material.json').symlink_to(source)
        with self.assertRaises(ValueError):
            self.load()

    def test_draft_training_or_validation_rejected_even_with_matching_manifest(self):
        for purpose in ('TRAIN', 'VALID'):
            original = json.loads((self.directory / 'new-material.json').read_text())
            edited = json.loads(json.dumps(original))
            p = json.loads(edited[purpose][0]['provenance']); p['draft'] = True
            edited[purpose][0]['provenance'] = json.dumps(p)
            self.write('new-material.json', edited); self.refresh_manifest()
            with self.assertRaises(ValueError):
                self.load()
            self.write('new-material.json', original); self.refresh_manifest()

    def test_acceptance_draft_or_training_permission_cannot_enter_formal_receipt(self):
        original = json.loads((self.directory / 'acceptance.json').read_text())
        for change in ('draft', 'training_allowed'):
            edited = json.loads(json.dumps(original))
            if change == 'draft':
                p = json.loads(edited[0]['provenance']); p['draft'] = True; edited[0]['provenance'] = json.dumps(p)
            else:
                edited[0]['training_allowed'] = True
            self.write('acceptance.json', edited); self.refresh_manifest()
            with self.assertRaises(ValueError):
                self.load()
