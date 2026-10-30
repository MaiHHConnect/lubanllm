import copy
import unittest

from verify_router_counterexamples import beneficial, fresh_pool, operand_key, select_arms
from verify_router_budget_counts import integer_gold
from luban.retention import normalized_context


def fixture():
    rows = []
    for group, count, action in (("local", 12, "local_core"), ("long", 14, "teacher"), ("short", 2, "teacher")):
        for i in range(count):
            rows.append({"id": f"{group}-{i:02}", "purpose": "training", "source": "synthetic",
                         "provenance": {"split": "train", "program_verified": True,
                            "verification_evidence": "fabricated CPU fixture, not model evidence",
                            "route_action": action, "family_group": "fixture-" + group}})
    return rows, [r["id"] for r in rows[:12]] + [f"long-{i:02}" for i in range(12)]


class RouterCounterexampleTests(unittest.TestCase):
    def test_equal_size_and_quantity_control_only_replace_training_failures(self):
        rows, baseline = fixture()
        arms, audit = select_arms(rows, baseline)
        self.assertEqual([len(arms[n]) for n in arms], [24, 24, 26, 26])
        self.assertEqual(set(arms['replace_short24']) - set(baseline), {'short-00', 'short-01'})
        self.assertEqual(set(baseline) - set(arms['replace_short24']), {'long-10', 'long-11'})
        self.assertEqual(audit['long_failure_controls'], ['long-12', 'long-13'])
        for name in arms:
            self.assertTrue(set(baseline[:12]) <= set(arms[name]))

    def test_selector_refuses_heldout_and_missing_counterexamples(self):
        rows, baseline = fixture()
        polluted = copy.deepcopy(rows)
        polluted[-1]['provenance']['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            select_arms(polluted, baseline)
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            select_arms(rows[:-1], baseline)

    def test_fresh_operands_contexts_and_gold_remain_fixed_across_splits(self):
        first, second = fresh_pool([]), fresh_pool([])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        contexts, operands = set(), set()
        for row in first:
            check = row['provenance']['checker']
            self.assertEqual(row['answer'], str(integer_gold(check)))
            self.assertNotIn(normalized_context(row), contexts)
            self.assertNotIn(operand_key(check['operation'], check['args']), operands)
            contexts.add(normalized_context(row)); operands.add(operand_key(check['operation'], check['args']))
        avoided = fresh_pool([first[0]])
        self.assertNotIn(normalized_context(first[0]), {normalized_context(r) for r in avoided})
        key = operand_key(first[0]['provenance']['checker']['operation'], first[0]['provenance']['checker']['args'])
        self.assertNotIn(key, {operand_key(r['provenance']['checker']['operation'], r['provenance']['checker']['args']) for r in avoided})

    def test_fewer_false_local_at_cost_of_all_correct_local_is_not_a_gain(self):
        baseline = {'route_accuracy': .75, 'effective_unsafe_local': 6, 'effective_correct_local': 6}
        all_teacher = {'route_accuracy': .8, 'effective_unsafe_local': 0, 'effective_correct_local': 0}
        useful = {'route_accuracy': .8, 'effective_unsafe_local': 4, 'effective_correct_local': 6}
        self.assertFalse(beneficial(all_teacher, baseline))
        self.assertTrue(beneficial(useful, baseline))
        self.assertFalse(beneficial(baseline, baseline))


if __name__ == '__main__':
    unittest.main()
