import unittest

import numpy as np

from probe_q4_paired_tail import tail_parts
from trace_q4_divergence import WIDTHS


class PairedTailTests(unittest.TestCase):
    def test_only_final_layer_exact_named_segments(self):
        raw = np.zeros((24, sum(WIDTHS.values())), np.float32)
        offset = 0
        for value, (name, width) in enumerate(WIDTHS.items(), 1):
            raw[22, offset:offset+width] = -value
            raw[23, offset:offset+width] = value
            offset += width
        actual = tail_parts(raw.reshape(-1))
        self.assertEqual(set(actual), {'h', 'act', 'down'})
        for name, expected in [('h', 1), ('act', 6), ('down', 7)]:
            np.testing.assert_array_equal(actual[name], np.full(WIDTHS[name], expected))
        raw[:] = 9
        self.assertEqual(actual['h'][0], 1)

    def test_missing_or_nonfinite_trace_rejected(self):
        with self.assertRaises(ValueError):
            tail_parts(np.zeros(sum(WIDTHS.values())))
        raw = np.zeros(24*sum(WIDTHS.values())); raw[0] = np.inf
        with self.assertRaises(ValueError):
            tail_parts(raw)


if __name__ == '__main__':
    unittest.main()
