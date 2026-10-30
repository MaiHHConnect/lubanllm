"""CPU row identity tests for the independent BF16 asymmetric reference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import capture_q4_task_reference as reference
from benchmark_native_decode import IDENTITY, MANIFEST
from prepare_q4_sequential import selected_inputs


class TaskReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old, self.task, self.out = [self.root/name for name in ('old', 'task', 'out')]
        for path in (self.old, self.task, self.out):
            path.mkdir()
        self.streams = patch.object(reference, 'STREAMS', {'layer00-gateup': 2048})
        self.streams.start(); self.addCleanup(self.streams.stop)
        self.plan = {'cases': [
            self.case('old-a', 3, 2, [0, 2, 3, 4]),
            self.case('old-b', 2, 1, [0, 1, 2]),
            self.case('observation', 2, 2, [0, 1, 2, 3], 'observation'),
            self.case('task-a', 4, 2, [0, 2, 3, 4, 5]),
            self.case('task-eos', 2, 0, [0, 1])]}  # EOS itself never enters a reference row.
        self.frames = {}
        for i, case in enumerate(self.plan['cases']):
            n = len(case['tokens']) + len(case['forced_tokens'])
            values = 0x3f00 + i*32 + np.arange(n, dtype=np.uint16)
            # Gate/up selected_inputs takes the second 2048 columns of kind-1 frames.
            self.frames[case['id']] = np.repeat(values[:, None], 4096, axis=1)

    @staticmethod
    def case(identifier, prompt, forwarded, positions, split='calibration'):
        return {'id': identifier, 'split': split, 'tokens': [1]*prompt,
                'forced_tokens': [2]*forwarded, 'positions': positions}

    def selected(self, plan):
        reply = {'records': [{'id': c['id']} for c in plan['cases']]}
        with patch('prepare_q4_sequential.read_frame', side_effect=lambda r, *a: self.frames[r['id']]):
            return selected_inputs(reply, plan, 0, 1, self.root)

    def write_sources(self):
        for directory, cases in ((self.old, self.plan['cases'][:3]), (self.task, self.plan['cases'][3:])):
            subplan = {'cases': cases}
            mapping = reference.layout(subplan)
            (directory/'position-mapping.json').write_text(json.dumps(mapping))
            # A task-only source has no observation cases; selected_inputs expects both splits.
            for split in reference.SPLITS:
                raw = []
                for phase in ('prompt', 'assistant'):
                    for case in cases:
                        if case['split'] == split:
                            positions = [p for p in case['positions']
                                         if (p < len(case['tokens'])) == (phase == 'prompt')]
                            raw.extend(self.frames[case['id']][positions, 2048:])
                np.asarray(raw, dtype='<u2').tofile(directory/f'layer00-gateup-{split}.bf16')

    def test_merge_equals_real_selected_inputs_not_whole_stream_concatenation(self):
        self.write_sources()
        before = {p: p.read_bytes() for directory in (self.old, self.task) for p in directory.iterdir()}
        mapping = reference.merge_streams(self.plan, 3, self.old, self.task, self.out)
        expected = self.selected(self.plan)
        for split, raw in expected.items():
            actual = np.fromfile(self.out/f'layer00-gateup-{split}.bf16', dtype='<u2').reshape(-1, 2048)
            np.testing.assert_array_equal(actual, raw)
        self.assertEqual((self.old/'layer00-gateup-observation.bf16').read_bytes(),
                         (self.out/'layer00-gateup-observation.bf16').read_bytes())
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
        cal = [r for r in mapping['rows'] if r['split'] == 'calibration']
        self.assertEqual([r['id'] for r in cal],
                         ['old-a']*2 + ['old-b']*2 + ['task-a']*3 + ['task-eos']*2
                         + ['old-a']*2 + ['old-b'] + ['task-a']*2)
        self.assertEqual([r['row'] for r in cal], list(range(len(cal))))
        for row in mapping['rows']:
            source = np.fromfile(Path(row['source_directory'])/f"layer00-gateup-{row['split']}.bf16",
                                 dtype='<u2').reshape(-1, 2048)
            self.assertEqual(source[row['source_row'], 0], self.frames[row['id']][row['position'], 2048])

    def test_rejects_source_reordering_missing_rows_nonfinite_and_wrong_size(self):
        self.write_sources()
        path = self.old/'position-mapping.json'
        mapping = json.loads(path.read_text())
        for change in ('reordered', 'missing', 'position'):
            altered = deepcopy(mapping)
            if change == 'reordered':
                altered['segments'][0], altered['segments'][1] = altered['segments'][1], altered['segments'][0]
            elif change == 'missing':
                altered['segments'].pop()
            else:
                altered['segments'][0]['positions'][0] = 1
            path.write_text(json.dumps(altered))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'row order'):
                reference.merge_streams(self.plan, 3, self.old, self.task, self.out)
        path.write_text(json.dumps(mapping))
        stream = self.old/'layer00-gateup-calibration.bf16'
        raw = stream.read_bytes()
        stream.write_bytes(raw[:-2])
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            reference.merge_streams(self.plan, 3, self.old, self.task, self.out)
        stream.write_bytes(b'\x80\x7f'+raw[2:])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            reference.merge_streams(self.plan, 3, self.old, self.task, self.out)

    def test_duplicate_ids_and_forward_boundary_rejected(self):
        plan = deepcopy(self.plan)
        plan['cases'][-1]['id'] = plan['cases'][0]['id']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reference.layout(plan)
        plan = deepcopy(self.plan)
        plan['cases'][-1]['positions'].append(2)
        with self.assertRaisesRegex(ValueError, 'positions'):
            reference.layout(plan)

    def test_loader_binds_receipt_plan_source_rows_and_stage_finiteness(self):
        plan = {'eos_ids': [99], 'cases': [self.case(f'case-{i}', 1, 1, [0, 1], 'observation' if 16 <= i < 20 else 'calibration')
                          for i in range(36)]}
        task = self.out/'task-capture'; task.mkdir()
        receipt, source_receipt = {'binary': 'new'}, {'binary': 'original'}
        mapping = reference.merged_mapping(plan, 20, self.old, task)
        result = {'schema': reference.SCHEMA, 'completed': True, 'receipt': receipt, 'plan': plan,
                  'inference_precision': 'bf16', 'core': None, 'manifest': MANIFEST,
                  'all_frozen_tokens_equal': True, 'cache_reuse': False, 'seed': 42, 'temperature': 0,
                  'captured_task_records': 16, 'original_count': 20, 'task_trajectories_from': 'original-source',
                  'task_source_receipt': source_receipt, 'original_reference': str(self.old),
                  'rows_per_split': mapping['rows_per_split']}
        result['task_records'], result['records'] = [], []
        for case in plan['cases'][20:]:
            record = {'id': case['id'], 'split': 'calibration', 'prompt_tokens': [1],
                      'generated_tokens': [2, 99], 'replay_tokens': [1, 2], 'assistant_positions': [1],
                      'complete': True}
            result['task_records'].append(record)
            result['records'].append({'id': case['id'], 'result': {'event': 'result', 'id': case['id'],
                'tokens': [2, 99], 'prompt_tokens': 1, 'generated_tokens': 2, 'finish_reason': 'capture',
                'complete': True, 'capture_only': True, 'cached_prompt_tokens': 0}})
        loaded = {'event': 'loaded', 'base_identity': IDENTITY, 'manifest_sha256': MANIFEST,
                  'core_loaded': False, 'core_file_sha256': None, 'quantized_projections': 0,
                  'inference_precision': 'bf16', 'research_capture_inputs': True}
        for path, value in ((self.out/'result.json', result), (self.out/'position-mapping.json', mapping),
                            (task/'loaded.json', loaded)):
            path.write_text(json.dumps(value))
        for split, count in mapping['rows_per_split'].items():
            np.full((count, 2048), 0x3f80, dtype='<u2').tofile(self.out/f'layer00-gateup-{split}.bf16')
        bound = reference.load_reference(self.out, receipt, plan, 'original-source', source_receipt)
        values = reference.read_reference(bound, 0, 1, 'calibration', 64)
        np.testing.assert_array_equal(values, np.ones((64, 2048), dtype=np.float32))
        with self.assertRaises(ValueError):
            reference.load_reference(self.out, {'binary': 'wrong'}, plan)
        with self.assertRaises(ValueError):
            reference.load_reference(self.out, receipt, plan, 'candidate-source', source_receipt)
        altered = deepcopy(result)
        altered['records'][0]['result']['tokens'][0] = 3
        (self.out/'result.json').write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'differs from frozen'):
            reference.load_reference(self.out, receipt, plan)
        (self.out/'result.json').write_text(json.dumps(result))
        mapping['rows'][0]['source_row'] += 1
        (self.out/'position-mapping.json').write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, 'row provenance'):
            reference.load_reference(self.out, receipt, plan)
        path = self.out/'layer00-gateup-calibration.bf16'
        with path.open('r+b') as stream:
            stream.write(b'\xc0\x7f')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            reference.read_reference(bound, 0, 1, 'calibration', 64)
        with self.assertRaisesRegex(ValueError, 'row count'):
            reference.read_reference(bound, 0, 1, 'calibration', 63)


if __name__ == '__main__':
    unittest.main()
