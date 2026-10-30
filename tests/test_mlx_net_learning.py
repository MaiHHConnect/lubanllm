"""Admission recomputes net gains; training wins never count as held-out mastery."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from luban.mlx_learning import formal_report, NET_LEARNING_POLICY, NET_QUALITY_GATES


def fixture(*, old=0):
    def sample(identifier, purpose):
        return dict(id=identifier, prompt='question ' + identifier, answer='ok', purpose=purpose,
                    approved=True, training_allowed=True, source='human', provenance={})
    groups = {'train': [sample('t' + str(i), 'training') for i in range(16)],
              'valid': [sample('v' + str(i), 'validation') for i in range(2)],
              'protection': [sample('p' + str(i), 'validation') for i in range(old)]}
    all_rows = sum(groups.values(), [])
    def observed(row, *, candidate):
        passed = candidate or row['id'].startswith('p')
        return dict(id=row['id'], answer='ok' if passed else 'wrong', passed=passed, complete=True,
                    reference_used=False, tokens=[1 if passed else 2], input_tokens=[42, all_rows.index(row)],
                    route_action='neural_expert', source='candidate' if candidate else 'parent', branch_enabled=True)
    before = [observed(row, candidate=False) for row in all_rows]
    after = [observed(row, candidate=True) for row in all_rows]
    def loss(value, source):
        return {'mean': value, 'observations': [dict(id=row['id'], loss=value, source=source) for row in groups['valid']]}
    return dict(promoted=True, quality_checked=True, inference_engine='mlx-q4',
                base_identity='fixture', core_version='core-current', active_expert='learned-current', chosen_width=64,
                learning_policy=NET_LEARNING_POLICY, gates=dict.fromkeys(NET_QUALITY_GATES, True), records=groups,
                observations={'baseline': [before, copy.deepcopy(before)], 'parent': [[], []],
                              'candidate': [after, copy.deepcopy(after)]},
                route_evidence={'rows': [dict(id=row['id'], action='neural_expert', input_tokens=row['input_tokens']) for row in before]},
                loss={'baseline_validation': loss(1., 'parent'), 'validation': loss(.5, 'candidate')},
                learning_progress=dict(policy=NET_LEARNING_POLICY, gained_ids=['v0', 'v1'], forgotten_ids=[],
                                       retained_ids=[row['id'] for row in groups['protection']], mastered_ids=['v0', 'v1'], net_gain=2),
                knowledge_pool=copy.deepcopy(groups['valid'] + groups['protection']),
                relearning_records=[], review_cursor=old)


def forget(report, identifier, *, baseline=False):
    for part in report['observations']['baseline' if baseline else 'candidate']:
        row = next(row for row in part if row['id'] == identifier)
        row.update(answer='unfinished', passed=False, complete=False, tokens=[3])
    if not baseline:
        report['learning_progress']['forgotten_ids'].append(identifier)
        report['learning_progress']['retained_ids'].remove(identifier)
        report['learning_progress']['net_gain'] -= 1
        report['knowledge_pool'] = [row for row in report['knowledge_pool'] if row['id'] != identifier]
        report['relearning_records'].append(next(row for row in report['records']['protection'] if row['id'] == identifier))


class MLXNetLearningTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = SimpleNamespace(store=self.root, identity='fixture', active_core='core-current')

    def admit(self, report):
        raw = json.dumps(report).encode()
        name = 'learning-123456abcdef.json'
        (self.root / name).write_bytes(raw)
        proof = dict(report=name, report_sha256=hashlib.sha256(raw).hexdigest(), expert_id='learned-current', width=64)
        return formal_report(self.runtime, proof)

    def test_first_round_has_no_old_questions_and_can_adopt_partial_new_mastery(self):
        report = fixture()
        for part in report['observations']['candidate']:
            row = next(row for row in part if row['id'] == 'v1')
            row.update(answer='unfinished', passed=False, complete=False, tokens=[3])
        report['learning_progress'].update(gained_ids=['v0'], mastered_ids=['v0'], net_gain=1)
        report['knowledge_pool'] = [report['records']['valid'][0]]
        self.assertEqual(self.admit(report)['learning_progress']['net_gain'], 1)

    def test_two_new_one_forgotten_is_net_gain_even_if_old_baseline_already_failed(self):
        report = fixture(old=1)
        forget(report, 'p0', baseline=True)
        forget(report, 'p0')
        self.assertEqual(self.admit(report)['learning_progress']['net_gain'], 1)

    def test_forged_gates_cannot_override_zero_or_negative_net_gain(self):
        for old in (2, 3):
            report = fixture(old=old)
            for i in range(old):
                forget(report, 'p' + str(i))
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, '没有独立检验净增益'):
                self.admit(report)

    def test_training_success_is_not_held_out_gain(self):
        report = fixture()
        for part in report['observations']['candidate']:
            for row in part:
                if row['id'].startswith('v'):
                    row.update(answer='wrong', passed=False, tokens=[2])
        report['learning_progress'].update(gained_ids=['t0'], mastered_ids=[], net_gain=1)
        report['knowledge_pool'] = []
        with self.assertRaisesRegex(ValueError, '净增统计'):
            self.admit(report)

    def test_actual_answer_completeness_and_reference_are_rechecked(self):
        for mutation in ({'answer': 'wrong'}, {'complete': False}, {'reference_used': True}, {'passed': 1}):
            report = fixture()
            for part in report['observations']['candidate']:
                part[-1].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.admit(report)

    def test_cannot_fake_baseline_wrong_to_claim_existing_knowledge_as_gain(self):
        report = fixture()
        for part in report['observations']['baseline']:
            part[-1]['answer'] = 'ok'
        with self.assertRaisesRegex(ValueError, '实际闭卷结果'):
            self.admit(report)

    def test_incomplete_baseline_and_duplicate_candidate_are_rejected(self):
        report = fixture()
        report['observations']['baseline'][0].pop()
        with self.assertRaisesRegex(ValueError, '覆盖不完整'):
            self.admit(report)
        report = fixture()
        report['observations']['candidate'][0][-1] = report['observations']['candidate'][0][-2]
        with self.assertRaisesRegex(ValueError, '覆盖不完整'):
            self.admit(report)

    def test_two_passes_and_route_inputs_must_be_identical(self):
        for group, field, value in [('baseline', 'tokens', [19]), ('candidate', 'tokens', [19]),
                                    ('candidate', 'input_tokens', [19]), ('baseline', 'source', 'core')]:
            report = fixture()
            report['observations'][group][1][-1][field] = value
            with self.subTest(group=group, field=field), self.assertRaises(ValueError):
                self.admit(report)

    def test_core_route_cannot_produce_learning_improvement(self):
        report = fixture()
        report['route_evidence']['rows'][-1]['action'] = 'local_core'
        for key in ('baseline', 'candidate'):
            for part in report['observations'][key]:
                part[-1].update(route_action='local_core', source='core', branch_enabled=False)
        with self.assertRaisesRegex(ValueError, '核心路由不能冒充'):
            self.admit(report)

    def test_train_is_not_blocked_by_inference_route_but_valid_is(self):
        report = fixture()
        report['route_evidence']['rows'][0]['action'] = 'local_core'
        before = report['observations']['baseline'][0][0]
        before.update(route_action='local_core', source='core', branch_enabled=False)
        for group in ('baseline', 'candidate'):
            for part in report['observations'][group]:
                part[0] = copy.deepcopy(before)
        self.assertEqual(self.admit(report)['learning_progress']['net_gain'], 2)
        # A wrong held-out core answer cannot be bypassed by the TRAIN change.
        report['route_evidence']['rows'][-1]['action'] = 'local_core'
        before = report['observations']['baseline'][0][-1]
        before.update(route_action='local_core', source='core', branch_enabled=False)
        for group in ('baseline', 'candidate'):
            for part in report['observations'][group]:
                part[-1] = copy.deepcopy(before)
        with self.assertRaises(ValueError):
            self.admit(report)

    def test_loss_must_be_finite_improved_and_match_covered_observations(self):
        for mutation in ('nan', 'negative', 'worse', 'faked_mean', 'missing_row'):
            report = fixture()
            loss = report['loss']['validation']
            if mutation in ('nan', 'negative', 'worse'):
                loss['mean'] = {'nan': float('nan'), 'negative': -1., 'worse': 1.}[mutation]
            elif mutation == 'faked_mean':
                loss['observations'][0]['loss'] = 5.
            else:
                loss['observations'].pop()
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, '损失'):
                self.admit(report)

    def test_failed_review_cannot_remain_mastered_or_disappear_from_relearning(self):
        for keep_old in (True, False):
            report = fixture(old=1)
            forget(report, 'p0')
            if keep_old:
                report['knowledge_pool'].append(report['records']['protection'][0])
            else:
                report['relearning_records'] = []
            with self.subTest(keep_old=keep_old), self.assertRaisesRegex(ValueError, '本轮结果不一致'):
                self.admit(report)

    def test_held_out_context_or_template_family_cannot_leak_from_training(self):
        for leak in ("context", "family"):
            report = fixture()
            if leak == "context":
                report['records']['valid'][0]['prompt'] = report['records']['train'][0]['prompt']
            else:
                for group in ('train', 'valid'):
                    report['records'][group][0]['provenance']['split_family'] = 'same'
            with self.subTest(leak=leak), self.assertRaisesRegex(ValueError, '泄漏'):
                self.admit(report)

    def test_pool_cannot_change_check_to_claim_different_knowledge(self):
        report = fixture()
        report['knowledge_pool'][0]['provenance']['generation_check'] = {'type': 'exact', 'expected': 'other'}
        with self.assertRaisesRegex(ValueError, '内容变化'):
            self.admit(report)

    def test_missing_gate_or_unknown_policy_is_rejected(self):
        report = fixture()
        report['gates']['mlx_net_gain'] = False
        with self.assertRaises(ValueError):
            self.admit(report)
        report = fixture()
        report['learning_policy'] = 'accept-anything'
        with self.assertRaisesRegex(ValueError, '策略未知'):
            self.admit(report)


if __name__ == '__main__':
    unittest.main()
