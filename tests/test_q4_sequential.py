"""CPU protocol and actual-input selection checks for sequential conversion."""
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
from prepare_q4_sequential import (FRAME_MAGIC, HEADER_BYTES, MANIFEST, SequentialWorker,
    read_frame, selected_inputs, stage_header, sparse_payloads, compare_control, verify_reload,
    protection_generations, regularization_source)
from benchmark_native_mixed import MixedWorker


class SequentialTests(unittest.TestCase):
    def test_regularization_requires_same_binding_exact_mask_and_protected_prior(self):
        from q4_sequential_protection import protection_cases, establish_baseline
        cases = protection_cases()
        outputs = [{'id': c['id'], 'text': c['check']['expected'] if isinstance(c['check']['expected'], str)
                    else json.dumps(c['check']['expected']), 'complete': True, 'tokens': [10, 11]} for c in cases]
        baseline = establish_baseline(cases, outputs)
        prior = {'completed': True, 'mode': 'convert', 'receipt': {'build': 'same'}, 'plan': {'cases': ['same']},
                 'reload': {'passed': True}, 'accepted_layers': [0, 23], 'layer_mask': 1+(1 << 23),
                 'artifact': {'quantized_projections': 5}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'accepted-generations.json').write_text(json.dumps(outputs))
            (root/'protection-baseline.json').write_text(json.dumps(baseline))
            def check(data):
                (root/'result.json').write_text(json.dumps(data))
                return regularization_source(root, prior['receipt'], prior['plan'])
            self.assertEqual(check(prior), ([0, 23], outputs))
            for key, value in [('completed', False), ('mode', 'control'), ('receipt', {}), ('plan', {}),
                               ('reload', {'passed': False}), ('accepted_layers', []), ('accepted_layers', [0, 0, 23]),
                               ('accepted_layers', [23, 0]), ('layer_mask', 1), ('artifact', {'quantized_projections': 6})]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError): check({**prior, key: value})
            outputs[0]['text'] = 'lost correct answer'
            (root/'accepted-generations.json').write_text(json.dumps(outputs))
            with self.assertRaisesRegex(ValueError, 'already lost'): check(prior)

    def fixture(self, root, name='cal-a', split='calibration', shift=0, kind=1):
        case = {'id': name, 'split': split, 'tokens': [1, 2], 'forced_tokens': [3], 'positions': [0, 1, 2]}
        width = 6144 if kind == 2 else 4096
        category = 'act' if kind == 2 else 'inputs'
        path = root/'layer-00'/category/(name+'.bin')
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.zeros((3, width), dtype='<u2')
        for row in range(3):
            raw[row] = 0x3f00+shift+row
        if kind == 1:
            raw[:, :2048] = 0x4000  # h must never enter the gate/up calibration stream.
        path.write_bytes(FRAME_MAGIC+struct.pack('<6I', 0, kind, 3, width, 2, 1)
                         +MANIFEST.encode()+b'0'*64+raw.tobytes())
        record = {'id': name, 'split': split, 'path': str(path), 'rows': 3, 'prompt_tokens': 2,
                  'forced_tokens': 1, 'row_values': width, 'kind': kind}
        return case, record, raw

    def test_actual_x_and_q4_act_keep_prompt_first_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in (1, 2):
                fixtures = [self.fixture(root, name, split, offset, kind) for name, split, offset in
                            [('cal-a', 'calibration', 0), ('obs-a', 'observation', 16), ('cal-b', 'calibration', 32)]]
                plan = {'cases': [v[0] for v in fixtures]}
                reply = {'records': [v[1] for v in fixtures]}
                selected = selected_inputs(reply, plan, 0, kind, root)
                width = 2048 if kind == 1 else 6144
                self.assertEqual(selected['calibration'].shape, (6, width))
                self.assertEqual(selected['calibration'][:, 0].tolist(),
                                 [0x3f00, 0x3f01, 0x3f20, 0x3f21, 0x3f02, 0x3f22])
                self.assertEqual(selected['observation'][:, 0].tolist(), [0x3f10, 0x3f11, 0x3f12])
                with self.assertRaisesRegex(ValueError, 'ordered'):
                    selected_inputs({'records': reply['records'][::-1]}, plan, 0, kind, root)

    def test_augmented_control_compares_original_streams_without_task_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = [self.fixture(root, name, split, shift) for name, split, shift in
                        [('cal-a', 'calibration', 0), ('observe-a', 'observation', 16),
                         ('seq-train-a', 'calibration', 32)]]
            original = {'cases': [f[0] for f in fixtures[:2]]}
            augmented = {'cases': [f[0] for f in fixtures]}
            reply = {'records': [f[1] for f in fixtures]}
            all_rows = selected_inputs(reply, augmented, 0, 1, root)
            retained = selected_inputs({'records': reply['records'][:2]}, original, 0, 1, root)
            self.assertEqual(len(all_rows['calibration']), 6)
            self.assertEqual(len(retained['calibration']), 3)
            self.assertEqual(retained['calibration'][:, 0].tolist(), [0x3f00, 0x3f01, 0x3f02])
            np.testing.assert_array_equal(all_rows['observation'], retained['observation'])
            for split, rows in retained.items():
                (root/f'layer00-gateup-{split}.bf16').write_bytes(rows.tobytes())
            self.assertTrue(all(c['exact'] for c in compare_control(retained, root, 0, 1)))
            with self.assertRaises(ValueError): compare_control(all_rows, root, 0, 1)

    def test_frame_rejects_wrong_binding_length_nonfinite_and_foreign_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, record, expected = self.fixture(root)
            path = Path(record['path']); original = path.read_bytes()
            np.testing.assert_array_equal(read_frame(record, case, 0, 1, root), expected)
            for offset in (0, 8, 12, 16, 20, 24, 28, 32, 96):
                raw = bytearray(original); raw[offset] ^= 1; path.write_bytes(raw)
                with self.subTest(offset=offset), self.assertRaises(ValueError):
                    read_frame(record, case, 0, 1, root)
            for raw in (original[:-2], original+b'xx'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): read_frame(record, case, 0, 1, root)
            raw = bytearray(original); struct.pack_into('<H', raw, HEADER_BYTES, 0x7f80); path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'nonfinite'): read_frame(record, case, 0, 1, root)
            path.write_bytes(original)
            with self.assertRaises(ValueError): read_frame({**record, 'path': str(root/'foreign')}, case, 0, 1, root)

    def test_stage_header_binds_revision_layer_projection_and_original_source(self):
        raw = stage_header(7, 3, 7)
        self.assertEqual(len(raw), 160)
        self.assertEqual(raw[:8], b'LQ4ST01\n')
        self.assertEqual(struct.unpack_from('<4I', raw, 8), (7, 3, 2, 16))
        self.assertEqual(raw[24:88], MANIFEST.encode())
        self.assertEqual(raw[88:152], b'0'*64)
        self.assertEqual(struct.unpack_from('<Q', raw, 152), (7,))
        for arguments in [(7, 3, 6), (24, 3, 24), (23, 4, 23), (1, 7, 1), (True, 3, 1)]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError): stage_header(*arguments)

    def test_control_detects_single_bf16_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {split: np.full((2, 2048), 0x3f80, dtype='<u2') for split in ('calibration', 'observation')}
            for split, raw in values.items(): (root/f'layer00-gateup-{split}.bf16').write_bytes(raw.tobytes())
            self.assertEqual(sum(r['values_compared'] for r in compare_control(values, root, 0, 1)), 8192)
            values['observation'][1, 7] += 1
            with self.assertRaisesRegex(ValueError, 'differs'): compare_control(values, root, 0, 1)

    def test_control_response_requires_id_operation_and_exact_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = object.__new__(SequentialWorker)
            worker.out = Path(tmp); worker.serial = 0
            class Process:
                stdin = io.StringIO()
            worker.process = Process()
            good = {'event': 'sequential', 'id': 'seq-0001-decide-2', 'op': 'decide', 'revision': 3,
                    'layer': 2, 'phase': 'ready'}
            worker.receive = lambda timeout: good
            self.assertEqual(worker.control('decide', 2, seq_accept=False), good)
            sent = json.loads(worker.process.stdin.getvalue())
            self.assertFalse(sent['seq_accept'])
            self.assertEqual(sent['seq_revision'], 2)
            for key, value in [('id', 'other'), ('op', 'down'), ('revision', 2), ('event', 'error'),
                               ('layer', 3), ('phase', 'candidate')]:
                worker.serial = 0; worker.receive = lambda timeout, key=key, value=value: {**good, key: value}
                with self.subTest(key=key), self.assertRaises(ValueError): worker.control('decide', 2, seq_accept=True)

    def test_sequential_worker_requires_original_core_and_excludes_other_modes(self):
        for kwargs in ({}, {'core': None, 'intervention': True}, {'core': None, 'trace_directory': Path('/tmp')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MixedWorker(Path('/missing'), 'sequential', Path('/missing'), None,
                            sequential_directory=Path('/tmp'), **kwargs)

    def test_sparse_payloads_never_include_rejected_layers_and_skip_final_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b'x'*(8+2048*6144*3//4)
            (root/'layer-23-gateup.bin').write_bytes(stage_header(23, 3, 23)+payload+payload)
            rows = list(sparse_payloads(root, [23]))
            self.assertEqual([(layer, kind) for layer, kind, _ in rows], [(23, 'gate'), (23, 'up')])
            with (root/'layer-23-gateup.bin').open('ab') as stream: stream.write(b'x')
            with self.assertRaisesRegex(ValueError, 'trailing'): list(sparse_payloads(root, [23]))

    def test_independent_reload_requires_same_outputs_and_releases_original_storage(self):
        from unittest.mock import MagicMock
        expected = [{'id': 'a', 'tokens': [4, 9], 'complete': True, 'text': 'ok', 'prompt_tokens': [1, 2]}]
        for scenario in ('pass', 'changed', 'wrong_count', 'retained_original', 'bad_exit'):
            with tempfile.TemporaryDirectory() as tmp:
                worker = MagicMock()
                worker.loaded = {'quantized_projections': 3, 'manifest_sha256': MANIFEST,
                                 'retained_original_weight_bytes': 0}
                worker.process.returncode = 0
                actual = expected
                if scenario == 'changed': actual = [{**expected[0], 'tokens': [5, 9]}]
                if scenario == 'wrong_count': worker.loaded['quantized_projections'] = 6
                if scenario == 'retained_original': worker.loaded['retained_original_weight_bytes'] = 20
                if scenario == 'bad_exit': worker.process.returncode = 1
                with patch('prepare_q4_sequential.MixedWorker', return_value=worker) as factory, \
                     patch('prepare_q4_sequential.protection_generations', return_value=actual):
                    call = lambda: verify_reload(Path('/binary'), Path(tmp), object(), [{}], object(), [9],
                                                 expected, {'quantized_projections': 3}, 10)
                    if scenario == 'pass':
                        self.assertTrue(call()['passed'])
                        self.assertIsNone(factory.call_args.kwargs['core'])
                        self.assertEqual(factory.call_args.kwargs['precision'], 'mixed-q4-g16-packed')
                    else:
                        with self.assertRaises((ValueError, RuntimeError)): call()
                    worker.close.assert_called_once()

    def test_generation_bounds_use_model_vocab_and_completion_uses_real_eos(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            worker = MagicMock(); worker.out = Path(tmp)
            tokenizer = MagicMock()
            tokenizer.__len__.return_value = 3
            tokenizer.apply_chat_template.return_value = [1, 2]
            tokenizer.decode.return_value = 'ok'
            case = {'id': 'case', 'messages': [{'role': 'user', 'content': 'hi'}]}
            result = {'id': 'check-case', 'prompt_tokens': 2, 'cached_prompt_tokens': 0,
                      'tokens': [5, 9], 'finish_reason': 'stop', 'complete': True}
            worker.request.return_value = result
            rows = protection_generations(worker, [case], tokenizer, [9], 'check', 10)
            self.assertTrue(rows[0]['complete'])
            from transformers import BatchEncoding
            tokenizer.apply_chat_template.return_value = BatchEncoding({'input_ids': [1, 2]})
            mapped = protection_generations(worker, [case], tokenizer, [9], 'check', 10)
            self.assertEqual(mapped[0]['prompt_tokens'], [1, 2])
            for changes in ({'tokens': [10, 9]}, {'complete': False}, {'tokens': [5, 8]}):
                worker.request.return_value = {**result, **changes}
                with self.assertRaises(ValueError):
                    protection_generations(worker, [case], tokenizer, [9], 'check', 10)


if __name__ == '__main__':
    unittest.main()
