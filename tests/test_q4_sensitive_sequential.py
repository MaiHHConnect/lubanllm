"""Diagnostic arithmetic and failed-worker cleanup without a model or GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_q4_sensitive_sequential as driver


class SensitiveSequentialTests(unittest.TestCase):
    def test_metrics_keep_weighted_and_ordinary_errors_distinct(self):
        weights = np.eye(2, dtype=np.float32)
        fp = np.array([[4., 2.], [1., 1.]], np.float32)
        x = np.array([[2., 1.], [1., 1.]], np.float32)
        report = driver.output_metrics(weights, weights, weights*2, fp, x, np.array([3., .25]))
        self.assertEqual(report['old'], {'unweighted_sse': 5., 'weighted_sse': 15.})
        self.assertEqual(report['new'], {'unweighted_sse': 2., 'weighted_sse': .5})
        observation = driver.output_metrics(weights, weights, weights*2, fp, x)
        self.assertIsNone(observation['new']['weighted_sse'])
        with np.errstate(invalid='ignore'), self.assertRaisesRegex(ValueError, 'nonfinite'):
            driver.output_metrics(weights, weights*np.nan, weights, fp, x)

    def test_parent_lock_is_released_even_when_worker_cleanup_raises(self):
        worker, lock = Mock(), Mock()
        worker.loaded = {}
        worker.control.side_effect = ValueError('original execution failure')
        worker.close.side_effect = RuntimeError('cleanup failure')
        source = {'plan': {'cases': []}, 'receipt': {}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'new-run'
            with patch.object(driver, 'preflight', return_value=(source, [], Path('/binary'), {}, {})), \
                    patch.object(driver, 'read', return_value={'weight_map': {}}), \
                    patch.object(driver, 'ModelProcessLock', return_value=lock), \
                    patch.object(driver, 'SequentialWorker', return_value=worker):
                with self.assertRaisesRegex(RuntimeError, 'cleanup failure'):
                    driver.run(output)
            lock.close.assert_called_once_with()
            evidence = json.loads((output/'failure.json').read_text())
            self.assertIn('original execution failure', evidence['error'])
            self.assertFalse((output/'result.json').exists())


if __name__ == '__main__':
    unittest.main()
