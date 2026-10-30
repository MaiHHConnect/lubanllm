"""CPU-only native capture format, token-control and stream-layout checks."""
import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from capture_q4_native import (CaptureWorker, HEADER_BYTES, IDENTITY, MAGIC, MANIFEST, ORDER,
    ROW_VALUES, STREAMS, append_streams, assemble_streams, capture_positions, check_loaded, check_result,
    load_frozen, make_request, read_capture)
from capture_q4_decode import replay_record, MODEL
from prepare_q4_packed import materials


class NativeCaptureTests(unittest.TestCase):
    def record(self):
        case = {'id': 'cal-00', 'split': 'calibration', 'prompt_tokens': [1, 2, 3]}
        result = {'id': case['id'], 'tokens': [4, 5, 9], 'complete': True,
                  'finish_reason': 'stop', 'cached_prompt_tokens': 0, 'prompt_tokens': 3, 'generated_tokens': 3}
        return replay_record(case, result, [9], 10)

    def result(self, request):
        return {'event': 'result', 'id': request['id'], 'tokens': [4, 5, 9], 'complete': True,
                'finish_reason': 'capture', 'capture_only': True, 'cached_prompt_tokens': 0,
                'prompt_tokens': 3, 'generated_tokens': 3}

    def carrier(self, directory):
        request = make_request(self.record(), [9])
        rows = len(request['capture_positions'])
        event = {'event': 'capture', 'id': request['id'], 'path': str(directory/'cal-00.inputs.bin'),
                 'rows': rows, 'row_values': ROW_VALUES, 'positions': request['capture_positions'],
                 'prompt_tokens': 3, 'forced_tokens': 2}
        header = MAGIC+struct.pack('<6I', rows, ROW_VALUES, 3, 2, 24, 47)+MANIFEST.encode()+b'0'*64
        data = np.empty((rows, ROW_VALUES), dtype='<u2')
        at = 0
        for index, (_, width) in enumerate(ORDER):
            data[:, at:at+width] = np.arange(rows, dtype=np.uint16)[:, None]+index*16
            at += width
        Path(event['path']).write_bytes(header+np.array(event['positions'], dtype='<u4').tobytes()+data.tobytes())
        return request, event, data

    def test_prompt_and_decode_positions_include_all_short_and_no_final_output(self):
        record = self.record()
        self.assertEqual(capture_positions(record), [0, 1, 2, 3, 4])
        request = make_request(record, [9])
        self.assertEqual(request['forced_tokens'], [4, 5])
        self.assertEqual(request['max_tokens'], 3)
        self.assertFalse(request['cache_reuse'])
        long = {'prompt_tokens': list(range(938)), 'assistant_positions': list(range(938, 1074)),
                'replay_tokens': list(range(1193))}
        positions = capture_positions(long)
        self.assertLessEqual(len(positions), 288)
        self.assertTrue(set(range(922, 938)) <= set(positions))
        self.assertEqual(positions, sorted(set(positions)))
        for positions in ([3, 3], [4, 3], [5]):
            with self.assertRaises(ValueError):
                capture_positions({**record, 'assistant_positions': positions})

    def test_loaded_requires_native_bf16_original_core(self):
        good = {'event': 'loaded', 'base_identity': IDENTITY, 'manifest_sha256': MANIFEST,
                'core_loaded': False, 'core_file_sha256': None, 'quantized_projections': 0,
                'inference_precision': 'bf16', 'research_capture_inputs': True}
        check_loaded(good)
        for key, value in [('core_loaded', True), ('inference_precision', 'mixed-q4'),
                           ('manifest_sha256', 'bad'), ('research_capture_inputs', False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_loaded({**good, key: value})

    def test_exact_tokens_and_capture_completion_required(self):
        request = make_request(self.record(), [9])
        result = self.result(request)
        check_result(result, request, [4, 5, 9])
        for key, value in [('tokens', [4, 6, 9]), ('complete', False), ('capture_only', False),
                           ('finish_reason', 'stop'), ('cached_prompt_tokens', 3), ('tokens', [True, 5, 9])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_result({**result, key: value}, request, [4, 5, 9])

    def test_carrier_roundtrip_and_all47_streams_row_major(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, event, expected = self.carrier(root)
            data = read_capture(event, request, root)
            np.testing.assert_array_equal(data, expected)
            counts = {name: {'calibration': 0, 'observation': 0} for name in STREAMS}
            append_streams(data, 'observation', root, counts)
            at = 0
            self.assertEqual(len(ORDER), 47)
            for name, width in ORDER:
                got = np.fromfile(root/f'{name}-observation.bf16', dtype='<u2').reshape(len(data), width)
                np.testing.assert_array_equal(got, expected[:, at:at+width])
                self.assertEqual(counts[name], {'calibration': 0, 'observation': len(data)})
                self.assertFalse((root/f'{name}-calibration.bf16').exists())
                at += width
            self.assertEqual(at, ROW_VALUES)
            del data

    def test_assembly_matches_hf_prompt_first_order_with_position_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, event, original = self.carrier(root)
            second_record = copy.deepcopy(self.record())
            second_record['id'] = 'cal-01'
            second_event = {**event, 'id': 'cal-01', 'path': str(root/'cal-01.inputs.bin')}
            payload = Path(event['path']).read_bytes()
            second_data = original + np.uint16(1024)
            Path(second_event['path']).write_bytes(payload[:HEADER_BYTES+event['rows']*4]+second_data.tobytes())
            counts = {name: {'calibration': 0, 'observation': 0} for name in STREAMS}
            mapping = assemble_streams([self.record(), second_record], [event, second_event], root, root, counts, [9])
            expected = np.concatenate([original[:3], second_data[:3], original[3:], second_data[3:]])
            at = 0
            for name, width in ORDER:
                got = np.fromfile(root/f'{name}-calibration.bf16', dtype='<u2').reshape(10, width)
                np.testing.assert_array_equal(got, expected[:, at:at+width])
                self.assertEqual(counts[name], {'calibration': 10, 'observation': 0})
                at += width
            self.assertEqual([(s['id'],s['phase'],s['stream_row_start'],s['stream_row_end']) for s in mapping],
                             [('cal-00','prompt',0,3),('cal-01','prompt',3,6),
                              ('cal-00','assistant',6,8),('cal-01','assistant',8,10)])
            self.assertEqual([s['positions'] for s in mapping], [[0,1,2],[0,1,2],[3,4],[3,4]])
            saved = json.loads((root/'position-mapping.json').read_text())
            self.assertEqual(saved['segments'], mapping)

    def test_header_positions_source_size_nonfinite_and_path_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, event, _ = self.carrier(root)
            path = Path(event['path'])
            original = path.read_bytes()
            for offset in [0, 8, 12, 16, 20, 24, 28, 32, 96, HEADER_BYTES]:
                corrupt = bytearray(original); corrupt[offset] ^= 1
                path.write_bytes(corrupt)
                with self.subTest(offset=offset), self.assertRaises(ValueError):
                    read_capture(event, request, root)
            for corrupt in (original[:-2], original+b'xx'):
                path.write_bytes(corrupt)
                with self.assertRaises(ValueError): read_capture(event, request, root)
            corrupt = bytearray(original)
            struct.pack_into('<H', corrupt, HEADER_BYTES+4*event['rows'], 0x7f80)
            path.write_bytes(corrupt)
            with self.assertRaisesRegex(ValueError, 'nonfinite'): read_capture(event, request, root)
            path.write_bytes(original)
            with self.assertRaises(ValueError): read_capture({**event, 'path': str(root/'wrong.bin')}, request, root)
            saved = root/'saved.bin'; path.rename(saved); path.symlink_to(saved)
            with self.assertRaises(ValueError): read_capture(event, request, root)

    def test_frozen_sources_preserve_twenty_records_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'config.json').write_text(json.dumps({'vocab_size': 10, 'eos_token_id': 9}))
            authored = [{**row, 'tokens': 3} for row in materials()]
            source = {'schema': 'q4-packed-calibration-v1', 'completed': True, 'core': None,
                      'core_digest': '0'*64, 'manifest': MANIFEST, 'identity': IDENTITY, 'model': MODEL,
                      'materials': authored,
                      'counts': {name: {'calibration': 48, 'observation': 12} for name in STREAMS}}
            cases, records = [], []
            for row in authored:
                case = {'id': row['id'], 'split': row['split'], 'prompt_tokens': [1, 2, 3]}
                native = {**self.record()['native_result'], 'id': row['id']}
                cases.append(case); records.append(replay_record(case, native, [9], 10))
            protocol = {'schema': 'q4-decode-capture-v1', 'source_protocol': source, 'seed': 42,
                        'temperature': 0, 'cache_reuse': False, 'max_new_tokens': 256,
                        'cases': cases, 'eos_ids': [9], 'vocab_size': 10}
            generated = {'completed': True, 'records': records}
            (root/'protocol.json').write_text(json.dumps(protocol))
            (root/'generation-result.json').write_text(json.dumps(generated))
            with patch('capture_q4_native.MODEL', str(root)):
                _, got = load_frozen(root)
                self.assertEqual(got, records)
                for mutation in ('split', 'positions', 'generated', 'incomplete'):
                    bad = copy.deepcopy(generated)
                    if mutation == 'split': bad['records'][0]['split'] = 'observation'
                    elif mutation == 'positions': bad['records'][0]['assistant_positions'] = [3]
                    elif mutation == 'generated': bad['records'][0]['generated_tokens'][0] = 6
                    else: bad['completed'] = False
                    (root/'generation-result.json').write_text(json.dumps(bad))
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError): load_frozen(root)

    def test_worker_only_accepts_capture_then_result_for_same_request(self):
        request = make_request(self.record(), [9])
        with tempfile.TemporaryDirectory() as tmp:
            worker = CaptureWorker.__new__(CaptureWorker)
            worker.out = Path(tmp)
            worker.process = type('Process', (), {'stdin': io.StringIO()})()
            capture = {'event': 'capture', 'id': request['id']}
            result = self.result(request)
            for sequence, passed in [([capture, result], True), ([result], False),
                                     ([capture, capture, result], False),
                                     ([{'event': 'token', 'id': request['id'], 'token': 4}], False),
                                     ([{**capture, 'id': 'wrong'}], False)]:
                worker.receive = lambda timeout, it=iter(sequence): next(it)
                if passed:
                    self.assertEqual(worker.capture(request, [4, 5, 9]), (capture, result))
                else:
                    with self.assertRaises(ValueError): worker.capture(request, [4, 5, 9])


if __name__ == '__main__':
    unittest.main()
