"""Admission evidence must pair all modes and retain individual baseline wins."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_q4_residual import encode_bf16
from verify_q4_residual import summarize
from benchmark_native_mixed import MixedWorker


class ResidualEvidenceTests(unittest.TestCase):
    def rows(self):
        rows = []
        for mode in ('bf16', 'q4', 'rank8', 'rank32'):
            for phase in ('ranking', 'preservation', 'long'):
                for i in range(2):
                    rows.append({'mode': mode, 'phase': phase, 'case_id': str(i), 'tokens': [i],
                                 'score': {'passed': i == (0 if mode == 'bf16' else 1)}})
            rows += [{'mode': mode, 'phase': 'speed', 'generated_tokens': 96, 'decode_ns': 1_000_000_000,
                      'ttft_ns': 1_000_000, 'wall_seconds': 1.001} for _ in range(9)]
        return rows

    def test_gain_does_not_cancel_loss_and_missing_pair_rejected(self):
        rows = self.rows()
        result = summarize(rows)
        self.assertEqual(result['quality']['rank8']['preservation']['lost_baseline_correct'], ['0'])
        self.assertEqual(result['quality']['rank8']['preservation']['gained_correct'], ['1'])
        rows = [r for r in rows if not (r['mode'] == 'rank8' and r['phase'] == 'long')]
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_invalid_sidecar_rejected_before_opening_worker_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                MixedWorker(Path('/unused'), 'candidate_mixed', root, None, 'mixed-q4-g32-residual', Path('relative'))
            self.assertEqual(list(root.iterdir()), [])

    def test_bf16_factor_payload_and_nonfinite_rejected(self):
        result = np.frombuffer(encode_bf16(np.array([1., -2., 0.], dtype=np.float32)), dtype='<u2')
        np.testing.assert_array_equal(result, [0x3f80, 0xc000, 0])
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                encode_bf16(np.array([value], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
