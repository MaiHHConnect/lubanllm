"""CPU checks for rejecting invalid layer/paging acceptance evidence."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_layered_inference import compare_scores, exact_rows, exact_trace, same_bytes
import check_layered_inference as screen


class LayeredInferenceScreenTests(unittest.TestCase):
    def test_performance_flag_never_dispatches_quality_screen(self):
        with patch.object(screen, 'performance_only') as performance, patch.object(screen, 'main') as quality:
            screen.run(['--performance-only'])
            performance.assert_called_once_with()
            quality.assert_not_called()

    def test_performance_repeat_refuses_existing_output_without_starting_worker(self):
        prior = {'completed': True, 'binary': str(screen.BINARY), 'core': None,
                 'precision': 'mixed-q4-g16', 'reports': {
                     mode: {'screen_passed': True} for mode in ('new_full', 'paged_full')}}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'performance-idle').mkdir()
            with patch.object(screen, 'OUTPUT', Path(directory)), patch.object(screen, 'read', return_value=prior), \
                    patch.object(screen.os.path, 'ismount', return_value=True), patch.object(screen, 'BoundedWorker') as worker:
                with self.assertRaises(FileExistsError):
                    screen.performance_only()
                worker.assert_not_called()

    def test_gain_cannot_compensate_for_losing_a_current_q4_correct_answer(self):
        baseline = [{'id': 'a', 'score': {'passed': True}}, {'id': 'b', 'score': {'passed': False}}]
        candidate = [{'id': 'a', 'score': {'passed': False}}, {'id': 'b', 'score': {'passed': True}}]
        result = compare_scores(baseline, candidate)
        self.assertEqual(result['candidate_correct'], result['baseline_correct'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['lost_current_q4_correct_ids'], ['a'])
        self.assertEqual(result['gained_ids'], ['b'])
        self.assertTrue(compare_scores(baseline, baseline)['passed'])

    def test_missing_or_empty_correct_coverage_cannot_pass(self):
        row = {'id': 'a', 'score': {'passed': True}}
        for other in ([], [row, row], [{'id': 'b', 'score': {'passed': True}}]):
            with self.assertRaises(ValueError):
                compare_scores([row], other)
        with self.assertRaises(ValueError):
            compare_scores([{'id': 'a', 'score': {'passed': False}}], [row])

    def test_same_text_does_not_hide_token_or_completion_changes(self):
        old = {'id': 'a', 'prompt_tokens': [1], 'tokens': [2, 3], 'text': 'ok', 'complete': True}
        exact_rows([old], [old])
        for changed in ({**old, 'tokens': [2, 4]}, {**old, 'complete': False}):
            with self.assertRaises(ValueError):
                exact_rows([changed], [old])

    def test_trace_checks_all_bytes_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a.bf16', Path(directory)/'b.bf16'
            a.write_bytes(b'\x00\x3f\x01\x40'); b.write_bytes(a.read_bytes())
            same_bytes(a, b, 4)
            b.write_bytes(b'\x00\x3f\x02\x40')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                same_bytes(a, b, 4)
            b.write_bytes(b'\x00\x3f')
            with self.assertRaisesRegex(ValueError, 'complete regular'):
                same_bytes(a, b, 4)
        with self.assertRaisesRegex(ValueError, 'two complete'):
            exact_trace({'events': []}, {'events': []})


if __name__ == '__main__':
    unittest.main()
