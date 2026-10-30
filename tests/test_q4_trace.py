"""Trace analysis rejects malformed numerical evidence and preserves token indices."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trace_q4_divergence import first_difference, read_bf16, error_stats
from benchmark_native_mixed import MixedWorker


class Q4TraceTests(unittest.TestCase):
    def test_divergence_is_next_token_index_not_prefix_length_plus_one(self):
        self.assertEqual(first_difference([842, 1226, 19078], [842, 1226, 43]), 2)
        self.assertEqual(first_difference([32], [33]), 0)
        self.assertIsNone(first_difference([32, 2], [32, 2]))
        self.assertEqual(first_difference([32], [32, 2]), 1)

    def test_trace_size_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.bf16'
            np.array([0x3f80, 0xc000], dtype='<u2').tofile(path)
            np.testing.assert_array_equal(read_bf16(path, 2), [1, -2])
            with self.assertRaisesRegex(ValueError, 'wrong trace size'):
                read_bf16(path, 3)
            np.array([0x7fc0, 0x7f80], dtype='<u2').tofile(path)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                read_bf16(path, 2)

    def test_error_stats_and_tie_policy(self):
        a, b = np.array([3., 4.]), np.array([3., 0.])
        result = error_stats(a, b)
        self.assertEqual(result['relative_l2'], .8)
        self.assertEqual(result['different_values'], 1)
        # Same equal-logit policy as native greedy: smallest token ID wins.
        logits = np.array([1., 5., 5., 3.])
        self.assertEqual(int(np.argmax(logits)), 1)
        self.assertEqual(int(np.argsort(-logits, kind='stable')[0]), 1)

    def test_trace_path_validation_precedes_process_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'absolute directory'):
                MixedWorker(Path('/unused'), 'baseline_bf16', root, None, trace_directory=Path('relative'))
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
