import unittest

import numpy as np

from diagnose_q4_sensitive_graph import logit_metrics


class LogitMetricsTests(unittest.TestCase):
    def test_ties_and_whole_vocabulary_competitor(self):
        reference = np.array([5., 5., 4., 1.])
        actual = np.array([4., 5., 6., 4.])
        result = logit_metrics(reference, actual, [0, 1])
        self.assertEqual(result['top1'], 2)
        self.assertEqual(result['reference_top1_margin'], -2)
        self.assertEqual(result['reference_top1_ties'], [0, 3])
        self.assertEqual(result['nominated']['0']['rank'], 3)
        original = logit_metrics(reference, reference, [0, 1])
        self.assertEqual(original['top_ties'], [0, 1])
        self.assertEqual(original['top1'], 0)
        self.assertEqual(original['reference_top1_margin'], 0)

    def test_probability_and_centered_error_ignore_common_offset(self):
        reference = np.array([2., 0., -3.])
        result = logit_metrics(reference, reference+100, [0])
        self.assertEqual(result['logits_error']['rmse'], 100)
        self.assertAlmostEqual(result['centered_logits_error']['rmse'], 0)
        self.assertAlmostEqual(result['teacher_student_kl'], 0)
        moved = logit_metrics(reference, np.zeros(3), [0])
        p = np.exp(reference)/np.exp(reference).sum()
        self.assertAlmostEqual(moved['teacher_student_kl'], float(np.sum(p*np.log(3*p))))


if __name__ == '__main__':
    unittest.main()
