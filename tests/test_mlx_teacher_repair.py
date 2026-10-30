import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from repair_mlx_teacher_expert import select_material


class TeacherRepairTests(unittest.TestCase):
    def setUp(self):
        def row(prompt, purpose):
            return dict(prompt=prompt, answer='authorized identity', purpose=purpose,
                        approved=1, training_allowed=1, route_action='neural_expert', provenance={})
        self.prepared = {'splits': {'train': [row('A', 'training'), row('B', 'training')],
                                    'development': [row('C', 'validation')]}}
        self.teachers = [dict(id=p, prompt=p, answer='authorized identity',
                             messages=[{'role': 'system', 'content': 'answer leak'}]) for p in ['A', 'B']]

    def test_teacher_context_never_enters_closed_book_material(self):
        before = copy.deepcopy(self.prepared)
        train, valid = select_material(self.prepared, self.teachers)
        self.assertEqual(before, self.prepared)
        for row in train + valid:
            self.assertEqual(row['messages'], [{'role': 'user', 'content': row['prompt']}])

    def test_teacher_capture_cannot_reclassify_validation(self):
        self.teachers[0]['prompt'] = 'C'
        with self.assertRaisesRegex(AssertionError, 'Held-out'):
            select_material(self.prepared, self.teachers)

    def test_conflicting_teacher_label_rejected(self):
        self.teachers[0]['answer'] = 'wrong identity'
        with self.assertRaisesRegex(AssertionError, 'conflicts'):
            select_material(self.prepared, self.teachers)

    def test_historical_hint_or_split_leak_rejected(self):
        self.prepared['splits']['train'][0]['provenance'] = {'messages': self.teachers[0]['messages']}
        with self.assertRaisesRegex(AssertionError, 'bare original'):
            select_material(self.prepared, self.teachers)


if __name__ == '__main__':
    unittest.main()
