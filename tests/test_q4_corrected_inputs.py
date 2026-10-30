"""CPU-only scope, binary layout and phase ordering checks; no model execution."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import capture_q4_corrected_inputs as corrected
import capture_q4_native as original
from capture_q4_task_reference import layout


class CorrectedInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = {'id': 'sample', 'tokens': [10, 11], 'forced_tokens': [12],
                        'max_tokens': 2, 'capture_positions': [0, 1, 2]}

    def carrier(self, request, code=0x3f00):
        p, f, positions = len(request['tokens']), len(request['forced_tokens']), request['capture_positions']
        path = self.root/(request['id']+'.inputs.bin')
        raw = np.repeat((code+np.asarray(positions, dtype=np.uint16))[:, None], corrected.ROW_VALUES, axis=1)
        header = (corrected.MAGIC+struct.pack('<6I', len(positions), corrected.ROW_VALUES, p, f, 24, 47)
                  +corrected.MANIFEST.encode()+b'0'*64)
        path.write_bytes(header+np.asarray(positions, dtype='<u4').tobytes()+raw.tobytes())
        event = {'event': 'capture', 'id': request['id'], 'path': str(path), 'positions': positions,
                 'rows': len(positions), 'row_values': corrected.ROW_VALUES, 'prompt_tokens': p, 'forced_tokens': f}
        return event, raw

    def test_q4_magic_is_not_bf16_and_corruption_rejected(self):
        event, expected = self.carrier(self.request)
        np.testing.assert_array_equal(corrected.read_capture(event, self.request, self.root), expected)
        with self.assertRaisesRegex(ValueError, 'binding'):
            original.read_capture(event, self.request, self.root)
        path = Path(event['path']); good = path.read_bytes()
        for changed in (original.MAGIC+good[8:], good[:-2],
                        good[:160]+struct.pack('<I', 1)+good[164:],
                        good[:172]+b'\x80\x7f'+good[174:]):
            path.write_bytes(changed)
            with self.assertRaises(ValueError):
                corrected.read_capture(event, self.request, self.root)
        path.write_bytes(good)
        with self.assertRaises(ValueError):
            corrected.read_capture({**event, 'id': 'other'}, self.request, self.root)

    def test_actual_predictions_may_differ_from_frozen_and_contain_early_eos(self):
        # Forced continuation continues even when packed greedy predicts an EOS.
        result = {'event': 'result', 'id': 'sample', 'tokens': [9, 8], 'prompt_tokens': 2,
                  'generated_tokens': 2, 'complete': True, 'capture_only': True,
                  'finish_reason': 'capture', 'cached_prompt_tokens': 0}
        corrected.checked_prediction(result, self.request, 20)
        for change in ({'tokens': [9]}, {'tokens': [9, 20]}, {'id': 'other'}, {'complete': 1},
                       {'cached_prompt_tokens': 1}, {'prompt_tokens': 3}, {'capture_only': False}):
            with self.assertRaises(ValueError):
                corrected.checked_prediction({**result, **change}, self.request, 20)

    def test_mapping_all_prompts_then_assistants_and_no_unforwarded_eos(self):
        requests = [self.request, {**self.request, 'id': 'task', 'tokens': [20, 21, 22],
                                  'forced_tokens': [], 'max_tokens': 1, 'capture_positions': [0, 2]},
                    {**self.request, 'id': 'obs'}]
        plan = {'cases': [{'id': r['id'], 'split': 'observation' if r['id'] == 'obs' else 'calibration',
                'tokens': r['tokens'], 'forced_tokens': r['forced_tokens'],
                'positions': r['capture_positions']} for r in requests]}
        events = [self.carrier(r, 0x3f00+i*16)[0] for i, r in enumerate(requests)]
        out = self.root/'inputs'; out.mkdir()
        mapping = corrected.assemble(plan, requests, events, self.root, out)
        self.assertEqual(mapping['rows_per_split'], {'calibration': 5, 'observation': 3})
        self.assertEqual([{k: s[k] for k in layout(plan)['segments'][0]} for s in mapping['segments']],
                         layout(plan)['segments'])
        for name, width in corrected.ORDER:
            values = np.fromfile(out/f'{name}-calibration.bf16', dtype='<u2').reshape(5, width)
            np.testing.assert_array_equal(values[:, 0], [0x3f00, 0x3f01, 0x3f10, 0x3f12, 0x3f02])
        bad = deepcopy(requests); bad[-1]['forced_tokens'] = []
        with self.assertRaises(ValueError):
            corrected.assemble(plan, bad, events, self.root, out)

    def test_exact_scope_excludes_thirteen_temporary_rejected_downs(self):
        names = corrected.exact_stream_names(corrected.LAYERS)
        self.assertEqual(len(names), 34)
        self.assertEqual(sum(n.endswith('gateup') for n in names), 24)
        self.assertEqual(sum(n.endswith('down') for n in names), 10)
        self.assertNotIn('layer01-down', names)
        self.assertNotIn('layer23-down', names)
        # A rejected layer gate/up stream is final-prefix input; its down stream is not.
        self.assertIn('layer01-gateup', names)
        self.assertEqual(len(set(n for n, _ in corrected.ORDER)-set(names)), 13)

    def test_trace_stream_extraction_exact_offsets_and_last_layer(self):
        raw = np.zeros((24, sum(corrected.WIDTHS.values())), dtype=np.uint16)
        for layer in range(24):
            at = 0
            for i, width in enumerate(corrected.WIDTHS.values()):
                raw[layer, at:at+width] = 0x3f00+layer*8+i
                at += width
        selected = corrected.trace_inputs(raw)
        self.assertEqual(selected.shape, (corrected.ROW_VALUES,))
        at = 0
        for name, width in corrected.ORDER:
            field = 2 if name.endswith('gateup') else 5
            self.assertTrue(np.all(selected[at:at+width] == 0x3f00+int(name[5:7])*8+field))
            at += width
        with self.assertRaises(ValueError):
            corrected.trace_inputs(raw.astype(np.float32))
        positions = self.request['capture_positions']
        self.assertEqual([positions.index(len(self.request['tokens'])-1+step) for step in (0, 1)], [1, 2])

    def test_loaded_scope_cannot_claim_ordinary_capture_or_bf16(self):
        loaded = {'event': 'loaded', 'base_identity': corrected.IDENTITY, 'manifest_sha256': corrected.MANIFEST,
            'core_loaded': False, 'core_file_sha256': None, 'retained_original_weight_bytes': 0,
            'inference_precision': 'mixed-q4-g16-packed', 'quantized_projections': 32,
            'packed_version': 3, 'residual_rank': 8, 'active_residual_projections': 32,
            'research_capture_inputs': True, 'research_intervention': False, 'research_sequential': False}
        corrected.checked_loaded(loaded, 3, 32, capture=True)
        for change in ({'research_capture_inputs': False}, {'inference_precision': 'bf16'},
                       {'packed_version': 2}, {'retained_original_weight_bytes': 1}, {'research_intervention': True}):
            with self.assertRaises(ValueError):
                corrected.checked_loaded({**loaded, **change}, 3, 32, capture=True)
        with self.assertRaises(ValueError):
            corrected.checked_loaded(loaded, 3, 32, capture=False)


if __name__ == '__main__':
    unittest.main()
