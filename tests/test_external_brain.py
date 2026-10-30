import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from luban.external_brain import ExternalBrain
from luban.memory import Memory

IDENTITY = 'a' * 64
BINDING = {'core_version': None, 'core_sha256': 'b' * 64, 'expert_id': None, 'expert_sha256': None}
CONTEXT = [{'role': 'system', 'content': '请按本轮虚构规程回答。'}, {'role': 'user', 'content': '领取日是哪天？'}]


class ExternalBrainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.memory = Memory(Path(self.temporary.name))
        self.memory.model_identity = IDENTITY
        self.brain = ExternalBrain(self.memory)

    def sample(self, *, context=None, answer='周三', source='human', extra=None):
        context = context or CONTEXT
        provenance = {'messages': context, 'base_identity': IDENTITY, **(extra or {})}
        return self.memory.add_sample(context[-1]['content'], answer, source=source, approved=True,
                                     training_allowed=True, provenance=provenance)

    def register(self, **kwargs):
        identifier = self.sample(**kwargs)
        result = self.brain.register_reference(identifier, kwargs.get('context', CONTEXT), IDENTITY, BINDING)
        return identifier, result

    def observe(self, answer='周三', **kwargs):
        defaults = {'complete': True, 'actual_context': CONTEXT, 'reference_used': False, 'local_action': 'local_core'}
        defaults.update(kwargs)
        return self.brain.observe_local(CONTEXT, IDENTITY, BINDING, answer, **defaults)

    def test_teacher_is_pending_never_mints_verified_or_training_sample(self):
        result = self.brain.record_teacher(CONTEXT, IDENTITY, '老师说周二',
            {'program_verified': True, 'human_verified': True, 'verification_evidence': 'fake', 'provider': 'fixture'})
        self.assertEqual(result['status'], 'needs_teacher')
        self.assertFalse(result['verified_source'] or result['training_allowed'])
        self.assertEqual(self.memory.samples(), [])
        material = self.memory.teacher_materials()[0]
        self.assertEqual(material['status'], 'pending')
        self.assertIsNone(material['sample_id'])
        provenance = json.loads(material['provenance'])
        self.assertNotIn('program_verified', provenance)
        self.assertNotIn('human_verified', provenance)
        self.assertNotIn('verification_evidence', provenance)
        self.assertIsNone(self.brain.lookup(CONTEXT, IDENTITY, BINDING)['reference'])

    def test_exact_context_role_history_and_unicode_must_match(self):
        _, result = self.register()
        self.assertEqual(result['status'], 'reference_ready')
        self.assertEqual(result['reference']['prompt'], CONTEXT[-1]['content'])
        self.assertTrue(result['external_reference'])
        variants = [CONTEXT[-1:], [{'role': 'system', 'content': '不同规程'}, CONTEXT[-1]],
                    CONTEXT[:1] + [{'role': 'user', 'content': '先前问题'}, {'role': 'assistant', 'content': '先前回答'}, CONTEXT[-1]],
                    CONTEXT[:-1] + [{'role': 'user', 'content': CONTEXT[-1]['content'] + ' '}]]
        for context in variants:
            result = self.brain.lookup(context, IDENTITY, BINDING)
            self.assertIsNone(result['reference'])
            self.assertEqual(result['status'], 'needs_teacher')
        wide = [{'role': 'user', 'content': 'Ａ'}]
        self.register(context=wide)
        self.assertIsNone(self.brain.lookup([{'role': 'user', 'content': 'A'}], IDENTITY, BINDING)['reference'])

    def test_unverified_teacher_wrong_context_wrong_base_and_route_only_rejected(self):
        for source, extra in [('teacher', {}), ('synthetic', {}), ('human', {'route_only': True})]:
            identifier = self.sample(source=source, extra=extra)
            with self.assertRaises(ValueError):
                self.brain.register_reference(identifier, CONTEXT, IDENTITY, BINDING)
        identifier = self.sample()
        with self.assertRaises(ValueError):
            self.brain.register_reference(identifier, [CONTEXT[-1]], IDENTITY, BINDING)
        with self.assertRaises(ValueError):
            self.brain.lookup(CONTEXT, 'c' * 64, BINDING)
        identifier = self.sample(source='teacher', extra={'program_verified': True, 'verification_evidence': 'Independent CPU rule fixture, not teacher self-rating'})
        self.assertEqual(self.brain.register_reference(identifier, CONTEXT, IDENTITY, BINDING)['status'], 'reference_ready')

    def test_only_real_original_context_complete_local_answer_mastered(self):
        identifier, _ = self.register()
        first = self.observe()
        self.assertEqual(first['status'], 'reference_ready')
        self.assertEqual(first['consecutive_passes'], 1)
        good = self.observe()
        self.assertEqual(good['status'], 'mastered')
        self.assertFalse(good['teacher_needed'])
        self.assertEqual(good['local_observation']['answer_sha256'], hashlib.sha256('周三'.encode()).hexdigest())
        self.assertEqual(self.brain.register_reference(identifier, CONTEXT, IDENTITY, BINDING)['status'], 'mastered')
        for kwargs in ({'complete': False}, {'reference_used': True}, {'local_action': 'teacher'},
                       {'actual_context': CONTEXT[:-1] + [{'role': 'user', 'content': CONTEXT[-1]['content'] + '\n参考答案：周三'}]}):
            self.observe()
            result = self.observe(**kwargs)
            self.assertEqual(result['status'], 'reference_ready')
            self.assertTrue(result['teacher_needed'])
            self.assertIsNotNone(result['reference'])
        self.assertFalse(self.observe(answer='周二')['verdict'])
        self.assertIsNone(self.observe(local_action='neural_expert')['verdict'])

    def test_core_hash_invalidates_mastery_but_selected_expert_does_not_erase_core_proof(self):
        self.register()
        self.observe()
        self.observe()
        changed = {**BINDING, 'core_sha256': 'd' * 64}
        result = self.brain.lookup(CONTEXT, IDENTITY, changed)
        self.assertEqual(result['status'], 'reference_ready')
        self.assertEqual(result['reference']['answer'], '周三')
        self.observe()
        self.observe()
        changed = {**BINDING, 'expert_id': 'learned-123456789abc', 'expert_sha256': 'e' * 64}
        self.assertEqual(self.brain.stats(IDENTITY, changed)['mastered'], 1)
        self.assertEqual(self.brain.lookup(CONTEXT, IDENTITY, changed)['status'], 'mastered')
        self.assertEqual(self.brain.lookup(CONTEXT, IDENTITY, changed)['local_action'], 'local_core')

    def test_engine_switch_invalidates_core_mastery_and_requires_two_new_executions(self):
        identifier = self.sample()
        hf = {**BINDING, 'inference': {'engine': 'torch-hf', 'torch_version': 'fixture'}}
        native = {**BINDING, 'inference': {'engine': 'luban-native-bf16-metal', 'binary_sha256': 'f' * 64}}
        self.brain.register_reference(identifier, CONTEXT, IDENTITY, hf)
        def observe(binding):
            return self.brain.observe_local(CONTEXT, IDENTITY, binding, '周三', complete=True,
                actual_context=CONTEXT, reference_used=False, local_action='local_core')
        observe(hf)
        self.assertEqual(observe(hf)['status'], 'mastered')
        changed = self.brain.lookup(CONTEXT, IDENTITY, native)
        self.assertEqual(changed['status'], 'reference_ready')
        self.assertEqual(changed['reference']['answer'], '周三')
        self.assertEqual(observe(native)['status'], 'reference_ready')
        self.assertEqual(observe(native)['status'], 'mastered')
        new_binary = {**native, 'inference': {**native['inference'], 'binary_sha256': 'e' * 64}}
        self.assertEqual(self.brain.lookup(CONTEXT, IDENTITY, new_binary)['status'], 'reference_ready')

    def test_legacy_proof_cannot_represent_native_execution(self):
        self.register()
        self.observe()
        self.observe()
        native = {**BINDING, 'inference': {'engine': 'luban-native-bf16-metal', 'binary_sha256': 'f' * 64}}
        self.assertEqual(self.brain.lookup(CONTEXT, IDENTITY, native)['status'], 'reference_ready')

    def test_delete_revoke_gold_or_provenance_change_invalidates_reference(self):
        for mutation in ('delete', 'revoke', 'gold', 'provenance'):
            with self.subTest(mutation=mutation):
                identifier, _ = self.register()
                self.observe()
                with self.memory.connect() as db:
                    if mutation == 'delete':
                        db.execute('DELETE FROM samples WHERE id=?', (identifier,))
                    elif mutation == 'revoke':
                        db.execute('UPDATE samples SET training_allowed=0 WHERE id=?', (identifier,))
                    elif mutation == 'gold':
                        db.execute("UPDATE samples SET answer='周二' WHERE id=?", (identifier,))
                    else:
                        db.execute("UPDATE samples SET provenance=json_set(provenance,'$.verification_evidence','changed') WHERE id=?", (identifier,))
                result = self.brain.lookup(CONTEXT, IDENTITY, BINDING)
                self.assertEqual(result['status'], 'needs_teacher')
                self.assertIsNone(result['reference'])
                self.assertTrue(result['teacher_needed'])

    def test_finite_program_rule_vs_advisory_and_full_output(self):
        context = [{'role': 'user', 'content': '虚构记录总数为三，只给整数'}]
        identifier = self.sample(context=context, answer='3', source='synthetic', extra={
            'program_verified': True, 'verification_evidence': 'Fixture literal count=3',
            'generation_check': {'type': 'semantic', 'validator': 'integer', 'expected': 3}})
        self.brain.register_reference(identifier, context, IDENTITY, BINDING)
        def observe(answer):
            return self.brain.observe_local(context, IDENTITY, BINDING, answer, complete=True,
                actual_context=context, reference_used=False, local_action='local_core')
        self.assertEqual(observe('3')['status'], 'reference_ready')
        self.assertEqual(observe('3')['status'], 'mastered')
        self.assertEqual(observe('3，但是其实是4')['status'], 'reference_ready')
        for check in ({'type': 'contains', 'expected': '周三'}, {'type': 'semantic', 'validator': 'nonempty'}):
            identifier = self.sample(extra={'generation_check': check})
            self.brain.register_reference(identifier, CONTEXT, IDENTITY, BINDING)
            result = self.observe()
            self.assertIsNone(result['verdict'])
            self.assertEqual(result['method'], 'advisory_only')
            self.assertTrue(result['teacher_needed'])

    def test_conflicting_approved_reference_is_not_latest_wins(self):
        first, _ = self.register()
        second = self.sample(answer='周二')
        with self.assertRaisesRegex(ValueError, '参考冲突'):
            self.brain.register_reference(second, CONTEXT, IDENTITY, BINDING)
        self.assertEqual(self.brain.lookup(CONTEXT, IDENTITY, BINDING)['reference']['id'], first)
        self.memory.delete_sample(first)
        self.assertEqual(self.brain.register_reference(second, CONTEXT, IDENTITY, BINDING)['reference']['id'], second)

    def test_capacity_evicts_least_recent_hint_only_and_lookup_uses_point_queries(self):
        with patch('luban.external_brain.CAPACITY', 3), patch.object(self.memory, 'samples', side_effect=AssertionError('whole table load forbidden')):
            contexts = [[{'role': 'user', 'content': '问题' + str(i)}] for i in range(4)]
            sample_ids = []
            for context in contexts[:3]:
                sample_ids.append(self.sample(context=context))
                self.brain.record_teacher(context, IDENTITY, '待核验老师回答')
                self.brain.register_reference(sample_ids[-1], context, IDENTITY, BINDING)
            self.brain.lookup(contexts[0], IDENTITY, BINDING)  # keep most recently used
            result = self.brain.record_teacher(contexts[3], IDENTITY, '新的待核验老师回答')
            self.assertEqual(result['evicted_hints'], 1)
            self.assertIsNone(self.brain.lookup(contexts[1], IDENTITY, BINDING)['reference'])
            self.assertIsNotNone(self.brain.lookup(contexts[0], IDENTITY, BINDING)['reference'])
            counts = self.brain.stats(IDENTITY, BINDING)
            self.assertEqual(counts['total'], 3)
            self.assertEqual(counts['needs_teacher'], 1)
            self.assertEqual(counts['reference_ready'], 2)
            with self.memory.connect() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM samples').fetchone()[0], 3)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM teacher_materials').fetchone()[0], 4)

    def test_teacher_arrival_never_removes_trusted_reference_or_grants_mastery(self):
        self.register()
        self.observe()
        self.brain.record_teacher(CONTEXT, IDENTITY, '与正确答案冲突的老师输出')
        result = self.brain.lookup(CONTEXT, IDENTITY, BINDING)
        self.assertEqual(result['reference']['answer'], '周三')
        self.assertEqual(result['status'], 'reference_ready')
        self.assertTrue(result['teacher_needed'])

    def test_invalid_bindings_and_contexts_rejected(self):
        for binding in ({}, {**BINDING, 'expert_id': 'expert'}, {**BINDING, 'core_sha256': 'bad'}):
            with self.assertRaises(ValueError):
                self.brain.lookup(CONTEXT, IDENTITY, binding)
        for context in ([], [{'role': 'assistant', 'content': 'wrong ending'}], [{'role': 'user', 'content': 'x', 'hidden': 'extra'}], [{'role': 'user', 'content': 'x' * 70000}]):
            with self.assertRaises(ValueError):
                self.brain.lookup(context, IDENTITY, BINDING)

    def test_absent_hint_never_forces_teacher_or_mints_failure_from_ungraded_answer(self):
        result = self.brain.lookup(CONTEXT, IDENTITY, BINDING)
        self.assertFalse(result['has_hint'] or result['teacher_needed'])
        result = self.observe(answer='没有金标准无法判定')
        self.assertFalse(result['has_hint'] or result['teacher_needed'])
        self.assertIsNone(result['verdict'])
        self.assertEqual(self.brain.stats(IDENTITY, BINDING)['total'], 0)

    def test_discover_uses_complete_context_and_earliest_same_gold_without_merging_sources(self):
        wrong_context = [{'role': 'system', 'content': '不同规定'}, CONTEXT[-1]]
        self.sample(context=wrong_context, answer='周一')
        self.sample(source='teacher', answer='老师未经核验的断言')
        first = self.sample()
        self.sample(source='synthetic', extra={'program_verified': True, 'verification_evidence': 'literal fixture'})
        with patch.object(self.memory, 'samples', side_effect=AssertionError('no full history scan')):
            result = self.brain.discover_reference(CONTEXT, IDENTITY, BINDING)
        self.assertTrue(result['has_hint'])
        self.assertEqual(result['reference']['id'], first)
        self.assertEqual(result['reference']['source'], 'human')
        self.assertNotIn('verification_evidence', result['reference']['provenance'])

    def test_discover_conflicts_or_over_capacity_fail_without_registering(self):
        self.sample(answer='周二')
        self.sample(answer='周三')
        with self.assertRaisesRegex(ValueError, '互相冲突'):
            self.brain.discover_reference(CONTEXT, IDENTITY, BINDING)
        self.assertFalse(self.brain.lookup(CONTEXT, IDENTITY, BINDING)['has_hint'])
        with self.memory.connect() as db:
            db.execute('DELETE FROM samples')
        for _ in range(4):
            self.sample()
        with patch('luban.external_brain.CAPACITY', 3), self.assertRaisesRegex(ValueError, '上限'):
            self.brain.discover_reference(CONTEXT, IDENTITY, BINDING)
        self.assertFalse(self.brain.lookup(CONTEXT, IDENTITY, BINDING)['has_hint'])

    def test_real_expert_binding_can_master_and_reference_changes_in_stats_invalidate(self):
        identifier, _ = self.register()
        binding = {**BINDING, 'expert_id': 'learned-123456789abc', 'expert_sha256': 'e' * 64}
        result = self.brain.observe_local(CONTEXT, IDENTITY, binding, '周三', complete=True,
            actual_context=CONTEXT, reference_used=False, local_action='neural_expert')
        self.assertEqual(result['status'], 'reference_ready')
        result = self.brain.observe_local(CONTEXT, IDENTITY, binding, '周三', complete=True,
            actual_context=CONTEXT, reference_used=False, local_action='neural_expert')
        self.assertEqual(result['status'], 'mastered')
        with self.memory.connect() as db:
            db.execute('UPDATE samples SET approved=0 WHERE id=?', (identifier,))
        self.assertEqual(self.brain.stats(IDENTITY, binding)['mastered'], 0)
        result = self.brain.lookup(CONTEXT, IDENTITY, binding)
        self.assertTrue(result['teacher_needed'])
        self.assertIsNone(result['reference'])

    def test_expert_mastery_requires_exact_expert_hash_and_core_binding(self):
        self.register()
        binding = {**BINDING, 'expert_id': 'learned-123456789abc', 'expert_sha256': 'e' * 64}
        self.brain.observe_local(CONTEXT, IDENTITY, binding, '周三', complete=True,
            actual_context=CONTEXT, reference_used=False, local_action='neural_expert')
        changed = {**binding, 'expert_sha256': 'f' * 64}
        self.assertEqual(self.brain.lookup(CONTEXT, IDENTITY, changed)['status'], 'reference_ready')

    def test_rechecks_bounded_and_actual_failure_or_null_checked_once_per_binding(self):
        contexts = [[{'role': 'user', 'content': '复测' + str(i)}] for i in range(6)]
        for context in contexts:
            self.register(context=context)
        candidates = self.brain.recheck_candidates(IDENTITY, BINDING)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0]['context'], contexts[0])
        for candidate in candidates:
            result = self.brain.observe_local(candidate['context'], IDENTITY, candidate['binding'], '错答',
                complete=True, actual_context=candidate['context'], reference_used=False, local_action='local_core')
            self.assertFalse(result['verdict'])
            self.assertEqual(result['checked_binding'], BINDING)
            self.assertTrue(result['teacher_needed'])
        self.assertEqual(len(self.brain.recheck_candidates(IDENTITY, BINDING)), 2)
        changed = {**BINDING, 'core_sha256': 'f' * 64}
        self.assertEqual(len(self.brain.recheck_candidates(IDENTITY, changed)), 4)
        for limit in (0, 5, True):
            with self.assertRaises(ValueError):
                self.brain.recheck_candidates(IDENTITY, BINDING, limit)
        context = [{'role': 'user', 'content': '开放式复测'}]
        self.register(context=context, extra={'generation_check': {'type': 'semantic', 'validator': 'nonempty'}})
        result = self.brain.observe_local(context, IDENTITY, BINDING, '答案', complete=True,
            actual_context=context, reference_used=False, local_action='local_core')
        self.assertIsNone(result['verdict'])
        self.assertEqual(result['checked_binding'], BINDING)
        self.assertNotIn(context, [row['context'] for row in self.brain.recheck_candidates(IDENTITY, BINDING)])

    def test_open_book_does_not_mark_rechecked_and_reference_revision_resets_check(self):
        identifier, _ = self.register()
        self.observe(reference_used=True)
        self.assertEqual(len(self.brain.recheck_candidates(IDENTITY, BINDING)), 1)
        self.observe(answer='错误答案')
        self.assertEqual(self.brain.recheck_candidates(IDENTITY, BINDING), [])
        self.brain.register_reference(identifier, CONTEXT, IDENTITY, BINDING)
        self.assertEqual(self.brain.recheck_candidates(IDENTITY, BINDING), [])
        self.brain.record_teacher(CONTEXT, IDENTITY, '又一条pending老师答复')
        self.assertEqual(self.brain.recheck_candidates(IDENTITY, BINDING), [])
        with self.memory.connect() as db:
            db.execute("UPDATE samples SET provenance=json_set(provenance,'$.source_revision','new') WHERE id=?", (identifier,))
        self.brain.register_reference(identifier, CONTEXT, IDENTITY, BINDING)
        self.assertEqual(len(self.brain.recheck_candidates(IDENTITY, BINDING)), 1)

    def test_checked_columns_migrate_existing_hint_table_without_losing_hints(self):
        from luban.external_brain import _context
        digest, encoded = _context(CONTEXT, IDENTITY)
        with self.memory.connect() as db:
            db.execute('''CREATE TABLE external_brain(identity TEXT,context_sha TEXT,context_json TEXT,state TEXT,
                sample_id TEXT,sample_sha TEXT,mastered_binding TEXT,teacher_material_id TEXT,last_used REAL,observation_json TEXT,
                PRIMARY KEY(identity,context_sha))''')
            db.execute('INSERT INTO external_brain VALUES (?,?,?,?,?,?,?,?,?,?)',
                (IDENTITY, digest, encoded, 'needs_teacher', None, None, None, None, 1, None))
        result = self.brain.lookup(CONTEXT, IDENTITY, BINDING)
        self.assertTrue(result['has_hint'])
        self.assertTrue(result['teacher_needed'])
        self.assertIsNone(result['checked_binding'])

    def test_two_checks_survive_restart_but_failure_and_reference_reset_streak(self):
        self.register()
        self.assertEqual(self.observe()['consecutive_passes'], 1)
        self.brain = ExternalBrain(self.memory)
        self.assertEqual(self.observe()['status'], 'mastered')
        self.assertEqual(self.observe(answer='周二')['consecutive_passes'], 0)
        self.assertEqual(self.observe()['status'], 'reference_ready')
        self.assertEqual(self.observe(reference_used=True)['consecutive_passes'], 0)
        self.assertEqual(self.observe()['status'], 'reference_ready')
        self.assertEqual(self.observe()['status'], 'mastered')

    def test_first_pass_allows_second_probe_and_failure_stops_same_version_retries(self):
        self.register()
        self.observe()
        self.assertEqual(len(self.brain.recheck_candidates(IDENTITY, BINDING)), 1)
        self.observe(answer='周二')
        self.assertEqual(self.brain.recheck_candidates(IDENTITY, BINDING), [])

    def test_legacy_one_check_mastery_is_not_grandfathered(self):
        self.register()
        self.observe()
        self.observe()
        with self.memory.connect() as db:
            db.execute("UPDATE external_brain SET observation_json=json_remove(observation_json,'$.checks'), "
                       "check_json=json_remove(check_json,'$.consecutive_passes')")
        self.assertEqual(self.brain.lookup(CONTEXT, IDENTITY, BINDING)['status'], 'reference_ready')
        self.assertEqual(self.observe()['status'], 'reference_ready')
        self.assertEqual(self.observe()['status'], 'mastered')

    def test_binding_or_reference_revision_never_combines_first_pass(self):
        identifier, _ = self.register()
        self.observe()
        changed = {**BINDING, 'core_sha256': 'f' * 64}
        observe = lambda: self.brain.observe_local(CONTEXT, IDENTITY, changed, '周三', complete=True,
            actual_context=CONTEXT, reference_used=False, local_action='local_core')
        self.assertEqual(observe()['status'], 'reference_ready')
        with self.memory.connect() as db:
            db.execute("UPDATE samples SET provenance=json_set(provenance,'$.source_revision','changed') WHERE id=?",
                       (identifier,))
        self.brain.register_reference(identifier, CONTEXT, IDENTITY, changed)
        self.assertEqual(observe()['status'], 'reference_ready')
        self.assertEqual(observe()['status'], 'mastered')

    def test_existing_teacher_material_link_does_not_duplicate_archive_or_mint_truth(self):
        attribution = {'provider': 'https://fixture.invalid', 'model': 'fixture', 'request_id': 'one'}
        identifier = self.memory.capture_teacher(CONTEXT, '原教师答复', attribution, eligible=True)
        before = self.memory.teacher_materials()[0]
        result = self.brain.record_teacher(CONTEXT, IDENTITY, '原教师答复', attribution, material_id=identifier)
        self.assertEqual(result['teacher_material_id'], identifier)
        self.assertEqual(self.memory.teacher_materials(), [before])
        self.assertEqual(before['status'], 'eligible')  # structural eligibility is not verification
        self.assertIsNone(self.brain.lookup(CONTEXT, IDENTITY, BINDING)['reference'])
        self.assertIsNone(self.brain.discover_reference(CONTEXT, IDENTITY, BINDING)['reference'])
        with self.assertRaisesRegex(ValueError, '不匹配'):
            self.brain.record_teacher([CONTEXT[-1]], IDENTITY, '原教师答复', attribution, material_id=identifier)
        with self.assertRaisesRegex(ValueError, '不匹配'):
            self.brain.record_teacher(CONTEXT, IDENTITY, '被替换的答案', attribution, material_id=identifier)
        with self.assertRaisesRegex(ValueError, '不匹配'):
            self.brain.record_teacher(CONTEXT, IDENTITY, '原教师答复', {**attribution, 'provider': 'another'}, material_id=identifier)
        self.assertEqual(self.memory.teacher_materials(), [before])


if __name__ == '__main__':
    unittest.main()
