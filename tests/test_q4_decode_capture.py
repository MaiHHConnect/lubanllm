"""CPU-only guards for fixed decode calibration positions and split preservation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from capture_q4_decode import IDENTITY, LIMIT, MANIFEST, MODEL, STREAMS, replay_record, validate_source


class DecodeCaptureTests(unittest.TestCase):
    def case(self, split='calibration'):
        return {'id': 'fixed', 'split': split, 'prompt_tokens': [1, 2, 3]}

    def result(self, tokens, stopped=True):
        return {'id': 'fixed', 'tokens': tokens, 'complete': stopped,
                'finish_reason': 'stop' if stopped else 'length', 'cached_prompt_tokens': 0,
                'prompt_tokens': 3, 'generated_tokens': len(tokens)}

    def source(self):
        return {'schema': 'q4-packed-calibration-v1', 'completed': True, 'core': None, 'core_digest': '0'*64, 'manifest': MANIFEST,
                'identity': IDENTITY, 'model': MODEL,
                'materials': [{'id': str(i), 'text': 'fixed prompt',
                               'split': 'calibration' if i < 16 else 'observation'} for i in range(20)],
                'counts': {name: {'calibration': 911, 'observation': 222} for name in STREAMS}}

    def test_first_eos_adds_no_assistant_input(self):
        r = replay_record(self.case(), self.result([9]), [9], 10)
        self.assertEqual(r['replay_tokens'], [1, 2, 3])
        self.assertEqual(r['assistant_positions'], [])
        self.assertEqual(r['generated_tokens'], [9])

    def test_last_emitted_token_not_forwarded(self):
        r = replay_record(self.case(), self.result([4, 5, 9]), [9], 10)
        self.assertEqual(r['replay_tokens'], [1, 2, 3, 4, 5])
        self.assertEqual(r['assistant_positions'], [3, 4])
        self.assertNotIn(2, r['assistant_positions'])  # First output uses an existing prompt state.

    def test_length_capped_output_kept_without_semantic_filtering(self):
        r = replay_record(self.case(), self.result([4]*LIMIT, False), [9], 10)
        self.assertEqual(len(r['replay_tokens']), 3+LIMIT-1)
        self.assertFalse(r['complete'])
        self.assertEqual(len(r['assistant_positions']), len(set(r['assistant_positions'])))
        self.assertTrue(set(range(3+LIMIT-17, 3+LIMIT-1)) <= set(r['assistant_positions']))
        self.assertEqual((min(r['assistant_positions']), max(r['assistant_positions'])), (3, 257))

    def test_observation_split_never_promoted(self):
        r = replay_record(self.case('observation'), self.result([4, 9]), [9], 10)
        self.assertEqual(r['split'], 'observation')

    def test_cancelled_short_missing_eos_and_early_eos_rejected(self):
        bad = [self.result([]), self.result([4], False), self.result([9, 4, 9]),
               {**self.result([4, 9]), 'finish_reason': 'cancelled'},
               {**self.result([4, 9]), 'cached_prompt_tokens': 1}, self.result([True, 9]),
               self.result([10, 9]), self.result([4]*LIMIT+[9])]
        for r in bad:
            with self.subTest(record=r), self.assertRaises(ValueError):
                replay_record(self.case(), r, [9], 10)

    def test_source_split_and_binding_guard(self):
        source = self.source()
        validate_source(source)
        for key, value in (('core', 'adopted'), ('core_digest', '1'*64), ('manifest', 'bad'), ('completed', False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_source({**source, key: value})
        for mutation in ('split', 'duplicate', 'count', 'missing_stream'):
            changed = copy.deepcopy(source)
            if mutation == 'split':
                changed['materials'][-1]['split'] = 'calibration'
            elif mutation == 'duplicate':
                changed['materials'][-1]['id'] = '0'
            elif mutation == 'count':
                changed['counts'][next(iter(STREAMS))]['observation'] = -1
            else:
                changed['counts'].pop(next(iter(STREAMS)))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_source(changed)


if __name__ == '__main__':
    unittest.main()
