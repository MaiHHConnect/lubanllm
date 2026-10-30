"""Offline checks for mixed precision evidence scoring; no model or GPU load."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('benchmark_native_mixed', ROOT / 'scripts/benchmark_native_mixed.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class MixedBenchmarkTests(unittest.TestCase):
    def test_group_calibrated_worker_loads_its_profile(self):
        worker = (ROOT / 'native/luban/overlay/zig/src/luban_worker.zig').read_text()
        self.assertIn(
            'if (q4_g32_group_calibrated) clip_profile = try loadClipProfile',
            worker,
        )

    def test_semantic_code_and_completion(self):
        fixture = {'check': {'type': 'semantic', 'validator': 'python_upper'}}
        for answer in ('def upper(s):\n    return s.upper()', '```python\ndef convert(text):\n    return text.upper()\n```'):
            self.assertTrue(bench.score(answer, fixture, True)['passed'])
            self.assertFalse(bench.score(answer, fixture, False)['passed'])
        self.assertFalse(bench.score('def upper(s):\n    return s.lower()', fixture, True)['passed'])

    def test_exact_numeric_without_float_rounding(self):
        fixture = {'scoring': 'exact_number', 'answer': 9007199254740993}
        self.assertTrue(bench.score('9007199254740993', fixture, True)['passed'])
        self.assertFalse(bench.score('9007199254740992', fixture, True)['passed'])
        self.assertFalse(bench.score('The answer is 9007199254740993', fixture, True)['passed'])

    def test_permissive_checks_are_not_finite(self):
        fixture = {'check': {'type': 'semantic', 'validator': 'nonempty'}}
        self.assertEqual(bench.score('anything', fixture, True)['finite'], False)
        self.assertFalse(bench.score('', fixture, True)['passed'])

    def test_lost_answer_cannot_be_offset_by_gain(self):
        rows = [dict(worker=worker, phase=suite, temperature=0, case_id=case_id,
                     score={'passed': passed, 'finite': True})
                for suite in ('ranking', 'preservation')
                for worker, scores in [('baseline_bf16', [True, False]), ('candidate_mixed', [False, True])]
                for case_id, passed in zip(('a', 'b'), scores)]
        report = bench.quality_summary(rows)
        self.assertEqual(report['ranking-t0']['lost_finite_correct_ids'], ['a'])
        self.assertEqual(report['ranking-t0']['gained_correct_ids'], ['b'])
        with self.assertRaises(ValueError):
            bench.quality_summary(rows[:-1])


if __name__ == '__main__':
    unittest.main()
