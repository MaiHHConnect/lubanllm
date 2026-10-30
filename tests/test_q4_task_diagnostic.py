"""Rejected candidate diagnosis cannot become admission; CPU-only scope controls."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_q4_task_candidate import diagnostic_result, exact_generations, ordinary_loaded, IDENTITY, MANIFEST


class TaskDiagnosticTests(unittest.TestCase):
    def test_only_ordinary_freed_original_workers_allowed(self):
        event = {'inference_precision': 'mixed-q4-g16-packed', 'base_identity': IDENTITY,
                 'manifest_sha256': MANIFEST, 'core_loaded': False, 'retained_original_weight_bytes': 0,
                 'research_sequential': False, 'research_capture_inputs': False,
                 'research_intervention': False, 'quantized_projections': 32}
        ordinary_loaded(event, 'mixed-q4-g16-packed')
        for key, value in [('retained_original_weight_bytes', 1), ('core_loaded', True),
                           ('research_sequential', True), ('research_capture_inputs', True),
                           ('research_intervention', True), ('quantized_projections', 71)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                ordinary_loaded({**event, key: value}, 'mixed-q4-g16-packed')

    def test_reload_requires_all_raw_trajectories_not_only_scores(self):
        rows = [{'id': str(i), 'prompt_tokens': [i], 'tokens': [10], 'text': 'wrong', 'complete': True}
                for i in range(16)]
        exact_generations(rows, deepcopy(rows))
        for field, value in [('id', 'other'), ('prompt_tokens', [2]), ('tokens', [11]),
                             ('text', 'correct'), ('complete', False)]:
            altered = deepcopy(rows); altered[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                exact_generations(altered, rows)
        with self.assertRaises(ValueError): exact_generations(rows[:-1], rows)

    def test_raw_completion_and_exposed_scoring_keep_wrong_and_truncated_answers(self):
        spec = {'id': 'old', 'case': {'tokens': [1, 2], 'max_tokens': 3, 'eos_ids': [99]},
                'fixture': {'check': {'type': 'exact', 'expected': 'yes'}}}
        tokenizer = Mock(); tokenizer.decode.return_value = 'no'
        raw = {'tokens': [10, 99], 'cached_prompt_tokens': 0, 'prompt_tokens': 2,
               'generated_tokens': 2, 'complete': True, 'finish_reason': 'stop'}
        row = diagnostic_result(spec, raw, tokenizer, 100)
        self.assertTrue(row['complete']); self.assertFalse(row['score']['passed'])
        self.assertIs(row['native_result'], raw)
        truncated = {**raw, 'tokens': [10, 10, 10], 'generated_tokens': 3, 'complete': False, 'finish_reason': 'length'}
        tokenizer.decode.return_value = 'yes'
        self.assertFalse(diagnostic_result(spec, truncated, tokenizer, 100)['score']['passed'])
        for changes in ({'cached_prompt_tokens': 1}, {'tokens': [100, 99]}, {'tokens': [99, 10]},
                        {'generated_tokens': 1}, {'complete': False}, {'finish_reason': 'cancelled'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                diagnostic_result(spec, {**raw, **changes}, tokenizer, 100)


if __name__ == '__main__':
    unittest.main()
