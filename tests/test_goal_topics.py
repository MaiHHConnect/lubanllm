import unittest

from luban.generation_checks import evaluate_generation, is_finite_generation_check, validate_generation_check
from luban.goal_topics import evaluate_goal_topics


FAST = '我的设计借鉴快慢协作：快脑先判断问题怎样处理，慢脑负责深入思考。'
WEIGHTS = '表层学习用于快速吸收新经验，深层学习把成熟知识逐步巩固到模型核心权重。'
GOAL = '我希望通过持续学习和积累，让自己的能力逐步成长。'
MASTERY = '我希望把请教后会答变成自己独立掌握的本领。'
POLICY = '我的升级原则是检验新知识并保护旧能力，只有新旧检验都合格才保留更新，失败就回退到可信版本。'
TEACHER = '遇到不会的内容，我的目标是先向老师请教，再用核验过的经验学习并逐步巩固到核心权重。'
INDEPENDENT = '不看参考也能通过独立检验，才是把老师的指导变成我自己的本领；同时必须守住旧能力。'
GOOD = {'framework': GOAL + FAST + WEIGHTS + MASTERY + POLICY,
        'fast_slow': FAST + WEIGHTS, 'own_mastery': TEACHER + INDEPENDENT, 'policy': POLICY}


class GoalTopicsTests(unittest.TestCase):
    def check(self, topic, answer, complete=True):
        return evaluate_generation(answer, {'type': 'semantic', 'validator': 'goal_topics', 'expected': topic}, complete)

    def test_invalid_category_fails_closed(self):
        for expected in (True, False, None, {}, [], 0, 1.5, '', 'unknown', 'FRAMEWORK'):
            with self.subTest(expected=expected):
                check = {'type': 'semantic', 'validator': 'goal_topics', 'expected': expected}
                for function in (validate_generation_check, is_finite_generation_check):
                    with self.assertRaises(ValueError): function(check)
                with self.assertRaises(ValueError): evaluate_generation(POLICY, check, True)
                with self.assertRaises(ValueError): evaluate_goal_topics(POLICY, expected, True)

    def test_finite_complete_compositions_and_sentence_order(self):
        for topic, answer in GOOD.items():
            self.assertTrue(is_finite_generation_check({'type': 'semantic', 'validator': 'goal_topics', 'expected': topic}))
            self.assertTrue(self.check(topic, answer)['passed'])
            self.assertFalse(self.check(topic, answer, False)['passed'])
            reversed_answer = '。'.join(reversed(answer[:-1].split('。'))) + '。'
            self.assertTrue(self.check(topic, reversed_answer)['passed'])

    def test_wrong_class_and_missing_concepts(self):
        for topic, answer in {'framework': GOAL + FAST + WEIGHTS + POLICY,
                              'fast_slow': FAST, 'own_mastery': TEACHER,
                              'policy': TEACHER + INDEPENDENT}.items():
            self.assertFalse(self.check(topic, answer)['passed'])

    def test_loops_contradictions_unknown_text_and_keyword_soup(self):
        for topic, good in GOOD.items():
            for answer in (good + good, good + '旧能力可以不保护。', good + '失败仍然升级。',
                           good + '我已拥有无限智能。', good + '谢谢。', '**' + good + '**',
                           '快脑慢脑表层深层权重检验新知识旧能力合格失败。', good + '我不再请教老师。'):
                self.assertFalse(self.check(topic, answer)['passed'])

    def test_lengths_and_sentence_count_are_bounded(self):
        self.assertFalse(self.check('policy', POLICY + '字' * 513 + '。')['passed'])
        self.assertFalse(self.check('framework', '。'.join([GOAL[:-1]] * 9) + '。')['passed'])
        self.assertFalse(self.check('policy', POLICY[:-1])['passed'])

    def test_generic_forbidden_still_applies(self):
        result = evaluate_generation(POLICY, {'type': 'semantic', 'validator': 'goal_topics',
                                     'expected': 'policy', 'forbidden': ['升级']}, True)
        self.assertTrue(result['correct'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['forbidden_found'], ['升级'])


if __name__ == '__main__': unittest.main()
