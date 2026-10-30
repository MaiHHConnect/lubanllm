"""Quality admission rejects lost or incomplete observations independently of totals."""
import copy
import hashlib
import json
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_native_precision_product import fresh_cases, result_score, route_cases, summarize, isolated_runtime_io, readonly_product_state, capture_generation, packed_candidates, read_holdout, cases_for_run, evaluation_stages, FIXTURES
from luban.runtime import Runtime
from luban.memory import Memory


class NativePrecisionProductTests(unittest.TestCase):
    @staticmethod
    def stage_rows(rows, candidate, plan, *, lose=False):
        for phase, fixture, temperature in plan:
            passed = not (lose and fixture['id'] == 'old-correct')
            rows.append({'label': candidate['label'], 'precision': candidate['precision'], 'phase': phase,
                         'case_id': fixture['id'], 'temperature': temperature, 'tokens': [int(passed)],
                         'complete': True, 'score': {'passed': passed}})

    def test_regression_failure_never_schedules_any_holdout_and_keeps_all_modes(self):
        modes = [{'label': name, 'precision': name} for name in ('bf16', 'bad', 'good')]
        plan = [('fresh', {'id': 'old-correct'}, 0), ('holdout', {'id': 'unseen'}, 0)]
        rows, calls = [], []
        with tempfile.TemporaryDirectory() as directory:
            for stage, candidate, subset in evaluation_stages(plan, modes, rows, Path(directory)):
                self.assertEqual(stage, 'regression')
                self.assertEqual(subset, plan[:1])
                calls.append(candidate['label'])
                self.stage_rows(rows, candidate, subset, lose=candidate['label'] == 'bad')
            self.assertEqual(calls, ['bf16', 'bad', 'good'])
            self.assertFalse(any(row['phase'] == 'holdout' for row in rows))
            result = json.loads((Path(directory)/'regression-result.json').read_text())
            self.assertTrue(result['completed'])
            self.assertTrue(result['final_holdout_skipped'])
            self.assertFalse(result['final_holdout_allowed'])
            self.assertEqual(result['rejected_candidates'], ['bad'])
            self.assertEqual(result['results']['bad']['quality']['fresh-t0']['lost_baseline_correct'], ['old-correct'])
            self.assertEqual(result['answers'], 3)

    def test_final_holdout_starts_only_after_all_paired_regression_results_saved(self):
        modes = [{'label': name, 'precision': name} for name in ('bf16', 'candidate')]
        plan = [('fresh', {'id': 'old-correct'}, 0), ('holdout-regression', {'id': 'exposed'}, 0),
                ('holdout', {'id': 'unseen'}, 0)]
        rows, calls = [], []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for stage, candidate, subset in evaluation_stages(plan, modes, rows, output):
                if stage == 'holdout':
                    gate = json.loads((output/'regression-result.json').read_text())
                    self.assertTrue(gate['final_holdout_allowed'])
                    self.assertEqual(gate['answers'], 4)
                    self.assertEqual(subset, plan[-1:])
                else:
                    self.assertEqual(subset, plan[:-1])
                calls.append((stage, candidate['label']))
                self.stage_rows(rows, candidate, subset)
            self.assertEqual(calls, [('regression', 'bf16'), ('regression', 'candidate'),
                                     ('holdout', 'bf16'), ('holdout', 'candidate')])
            self.assertTrue(summarize(rows, plan, ['bf16', 'candidate'])['candidate']['quality_retained'])

    def test_incomplete_regression_cannot_unlock_final_holdout(self):
        modes = [{'label': name, 'precision': name} for name in ('bf16', 'candidate')]
        plan = [('fresh', {'id': 'old-correct'}, 0), ('holdout', {'id': 'unseen'}, 0)]
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            stages = evaluation_stages(plan, modes, rows, Path(directory))
            _, candidate, subset = next(stages)
            self.stage_rows(rows, candidate, subset)
            self.assertEqual(next(stages)[0], 'regression')
            with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                next(stages)
            self.assertFalse((Path(directory)/'regression-result.json').exists())

    def test_without_final_holdout_preserves_original_mode_order(self):
        modes = [{'label': name, 'precision': name} for name in ('bf16', 'candidate')]
        plan = [('fresh', {'id': 'old-correct'}, 0), ('holdout-regression', {'id': 'exposed'}, 0)]
        rows, calls = [], []
        with tempfile.TemporaryDirectory() as directory:
            for stage, candidate, subset in evaluation_stages(plan, modes, rows, Path(directory)):
                self.assertEqual(stage, 'all')
                self.assertIs(subset, plan)
                calls.append(candidate['label'])
                self.stage_rows(rows, candidate, subset, lose=candidate['label'] == 'candidate')
            self.assertEqual(calls, ['bf16', 'candidate'])
            self.assertFalse((Path(directory)/'regression-result.json').exists())

    def test_equal_totals_do_not_cancel_regressions(self):
        plan = [('fresh', {'id': name}, 0) for name in ('kept', 'lost', 'gained')]
        rows = []
        for mode, flags in [('bf16', [True, True, False]), ('mixed-q4-g16', [True, False, True])]:
            for (_, fixture, temp), passed in zip(plan, flags):
                rows.append({'precision': mode, 'phase': 'fresh', 'case_id': fixture['id'], 'temperature': temp,
                             'score': {'passed': passed}, 'tokens': [int(passed)], 'complete': True})
        result = summarize(rows, plan, ['bf16', 'mixed-q4-g16'])['mixed-q4-g16']
        self.assertFalse(result['quality_retained'])
        self.assertEqual(result['quality']['fresh-t0']['lost_baseline_correct'], ['lost'])
        for invalid in [rows[:-1], rows + [copy.deepcopy(rows[-1])]]:
            with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                summarize(invalid, plan, ['bf16', 'mixed-q4-g16'])

    def test_identity_must_be_complete_and_use_expert(self):
        fixture = route_cases()[0]
        text = fixture['check']['expected']
        self.assertTrue(result_score(text, True, {'local_action': 'neural_expert'}, fixture, 'routes')['passed'])
        self.assertFalse(result_score(text, False, {'local_action': 'neural_expert'}, fixture, 'routes')['passed'])
        self.assertFalse(result_score(text, True, {'local_action': 'local_core'}, fixture, 'routes')['passed'])
        ordinary = route_cases()[-1]
        result = result_score('irrelevant', False, {'local_action': 'local_core'}, ordinary, 'routes')
        self.assertTrue(result['passed'])
        self.assertIn('route-only', result['method'])

    def test_fresh_labels_and_structure_are_valid(self):
        cases = fresh_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({row['id'] for row in cases}), 12)
        for row in cases:
            expected = row['check']['expected']
            import json
            answer = json.dumps(expected) if isinstance(expected, dict) else expected
            self.assertTrue(result_score(answer, True, {}, row, 'fresh')['passed'], row['id'])
            self.assertFalse(result_score(answer, False, {}, row, 'fresh')['passed'], row['id'])
        self.assertEqual(len(route_cases()), 13)

    def test_engine_is_recorded_inside_generation_scope_before_unwind(self):
        for expert, depth, action in [(object(), 1, 'neural_expert'), (None, 1, 'local_core'), (None, 0, 'local_core')]:
            with self.subTest(action=action, depth=depth):
                runtime = SimpleNamespace(branch=SimpleNamespace(expert=expert), cold_backbone=SimpleNamespace(depth_count=depth))
                def identity(*, actual_action):
                    self.assertEqual(actual_action, action)
                    self.assertEqual(runtime.cold_backbone.depth_count, depth)
                    return {'engine': 'torch-hf' if actual_action == 'neural_expert' or runtime.cold_backbone.depth_count else 'native'}
                runtime.inference_identity = identity
                def generate(tokens, maximum, **kwargs):
                    # Model the caller's subsequent scope unwind; evidence must already be captured.
                    runtime.branch.expert = None
                    runtime.cold_backbone.depth_count = 0
                    return torch.tensor([[1, 2, 3]])
                captured = {}
                output = capture_generation(runtime, generate, captured, torch.tensor([[1, 2]]), 1)
                self.assertEqual(output.tolist(), [[1, 2, 3]])
                self.assertEqual(captured['engine']['engine'], 'torch-hf' if expert is not None or depth else 'native')
                self.assertEqual(captured['execution_hf_depth'], depth)
                self.assertEqual(captured['execution_action'], action)
                self.assertEqual(captured['tokens'], [3])

    def test_packed_labels_remain_distinct_from_actual_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'payload with spaces.bin'; path.write_bytes(b'fixture')
            modes = packed_candidates(['rtn=' + str(path), 'gptq=' + str(path)])
            self.assertEqual([mode['label'] for mode in modes], ['rtn', 'gptq'])
            self.assertTrue(all(mode['precision'] == 'mixed-q4-g16-packed' for mode in modes))
            for values in [['bf16=' + str(path)], ['rtn=' + str(path), 'rtn=' + str(path)], ['bad label=' + str(path)], ['rtn=relative']]:
                with self.assertRaises(ValueError):
                    packed_candidates(values)
        plan = [('fresh', {'id': 'case'}, 0)]
        rows = [{'label': label, 'precision': precision, 'phase': 'fresh', 'case_id': 'case', 'temperature': 0,
                 'score': {'passed': passed}, 'tokens': [int(passed)], 'complete': True}
                for label, precision, passed in [('bf16','bf16',True), ('rtn','mixed-q4-g16-packed',True), ('gptq','mixed-q4-g16-packed',False)]]
        summary = summarize(rows, plan, ['bf16','rtn','gptq'])
        self.assertTrue(summary['rtn']['quality_retained'])
        self.assertFalse(summary['gptq']['quality_retained'])

    def test_frozen_holdout_is_finite_unique_and_appended_last(self):
        source = Path(__file__).resolve().parents[1] / 'research/q4_packed_holdout_20261016.json'
        holdout = read_holdout(source)
        self.assertEqual(len(holdout), 20)
        for fixture in holdout:
            expected = fixture['check']['expected']
            text = json.dumps(expected) if isinstance(expected, (dict,list)) else expected
            self.assertTrue(result_score(text, True, {}, fixture, 'holdout')['passed'])
            self.assertFalse(result_score(text, False, {}, fixture, 'holdout')['passed'])
        # The extension preserves the existing plan verbatim and appends only final observations.
        import verify_native_precision_product as module
        fake = {'ranking': [({'id':'rank','max_tokens':10}, {'id':'rank','messages':[{'role':'user','content':'fixture'}], 'expected':'x'})],
                'preservation': [({'id':'preserve','max_tokens':10}, {'id':'preserve','messages':[{'role':'user','content':'other'}], 'expected':'y'})]}
        with patch.object(module, 'load_cases', return_value=fake):
            original = cases_for_run(Path('/unused'))
            extended = cases_for_run(Path('/unused'), source)
        self.assertEqual(extended[:-20], original)
        self.assertTrue(all(phase == 'holdout' and temp == 0 for phase, _, temp in extended[-20:]))
        payload = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'holdout.json'
            for mutation in ('duplicate', 'permissive', 'unfrozen'):
                changed = copy.deepcopy(payload)
                if mutation == 'duplicate': changed['cases'].append(copy.deepcopy(changed['cases'][0]))
                if mutation == 'permissive': changed['cases'][0]['check'] = {'type':'contains','expected':'x'}
                if mutation == 'unfrozen': changed['frozen'] = False
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): read_holdout(path)

    def test_old_holdout_regression_and_new_holdout_form242_disjoint_cases(self):
        import verify_native_precision_product as module
        root = Path(__file__).resolve().parents[1]
        old = root / 'research/q4_packed_holdout_20261016.json'
        new = root / 'research/q4_decode_holdout_20261016.json'
        def fixtures(directory, limit=None):
            result = {}
            for phase, size in [('ranking',48),('preservation',112)]:
                result[phase] = [({'id':f'{phase}-{index}', 'max_tokens':192},
                    {'id':f'{phase}-{index}', 'messages':[{'role':'user','content':f'fixture-{phase}-{index}'}],
                     'expected':'x'}) for index in range(size if limit is None else limit)]
            return result
        with patch.object(module, 'load_cases', side_effect=fixtures):
            baseline = cases_for_run(Path('/unused'))
            plan = cases_for_run(Path('/unused'), new, old)
            self.assertEqual(len(baseline),202)
            self.assertEqual(len(plan),242)
            self.assertEqual(plan[:202],baseline)
            self.assertTrue(all(phase=='holdout-regression' and temp==0 for phase,_,temp in plan[202:222]))
            self.assertTrue(all(phase=='holdout' and temp==0 for phase,_,temp in plan[222:]))
            self.assertEqual([row['id'] for _,row,_ in plan[202:222]], [row['id'] for row in read_holdout(old)])
            with self.assertRaisesRegex(ValueError,'overlaps'):
                cases_for_run(Path('/unused'), old, old)
            with tempfile.TemporaryDirectory() as directory:
                duplicate = json.loads(new.read_text())
                duplicate['cases'][0]['messages'] = copy.deepcopy(read_holdout(old)[0]['messages'])
                duplicate['cases'][0]['messages'][0]['content'] += '  '
                path = Path(directory)/'overlap.json';path.write_text(json.dumps(duplicate))
                with self.assertRaisesRegex(ValueError,'overlaps'):
                    cases_for_run(Path('/unused'), path, old)

    @staticmethod
    def make_journal(root, rows=()):
        database = root / 'app/memory.sqlite3'
        database.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(database)
        try:
            db.execute('CREATE TABLE training_runs (identity TEXT, body TEXT, state TEXT)')
            db.executemany('INSERT INTO training_runs VALUES (?,?,?)', rows)
            db.commit()
        finally:
            db.close()
        return database

    def test_real_runtime_load_keeps_private_memory_when_factory_recreates_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, storage = root / 'model', root / 'production'
            model.mkdir(); storage.mkdir()
            (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
            (model / 'upstream-file-manifest.json').write_text(json.dumps({'Data': {'Files': []}}))
            database = self.make_journal(storage)
            before = database.read_bytes()
            private = Memory(root / 'private')
            runtime = Runtime.__new__(Runtime)
            runtime.lock = threading.RLock()
            runtime.settings = SimpleNamespace(snapshot=lambda: {'model_path': str(model), 'storage_root': str(storage)})
            runtime.device = 'cpu'; runtime.dtype = None; runtime.research = False
            runtime._model_process_lock = None
            runtime._release_model = lambda **kwargs: None
            runtime.memory = Memory(storage)
            with isolated_runtime_io(private, storage / 'modules/luban0-v1'), \
                    patch('luban.runtime.ModelProcessLock'), \
                    patch('luban.runtime.load_backbone', side_effect=RuntimeError('stop before model allocation')):
                with self.assertRaisesRegex(RuntimeError, 'stop before model allocation'):
                    runtime.load()
            self.assertIs(runtime.memory, private)
            self.assertEqual(private.model_identity, hashlib.sha256(json.dumps([]).encode()).hexdigest())
            self.assertEqual(database.read_bytes(), before)
            self.assertFalse(private.path.exists())

    def test_readonly_admission_rejects_pending_and_unaccepted_active_research(self):
        for state in ('queued', 'running', 'paused', 'committing'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); store = root / 'modules/luban0-v1'; store.mkdir(parents=True)
                database = self.make_journal(root, [('identity', '{}', state)])
                before = database.read_bytes()
                with self.assertRaisesRegex(ValueError, 'pending training or recovery'):
                    readonly_product_state(store, 'identity')
                self.assertEqual(database.read_bytes(), before)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); store = root / 'modules/luban0-v1'; store.mkdir(parents=True)
            pointer = {'id': 'unaccepted'}
            (store / 'active.json').write_text(json.dumps(pointer))
            body = {'scope': 'research', 'commit': {'pointers': [{'name': 'active.json', 'after': pointer}]}}
            database = self.make_journal(root, [('identity', json.dumps(body), 'rejected')])
            before = database.read_bytes()
            with self.assertRaisesRegex(ValueError, '尚未通过产品验收'):
                readonly_product_state(store, 'identity')
            self.assertEqual(database.read_bytes(), before)
            readonly_product_state(store, 'another-identity')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'must exist'):
                readonly_product_state(Path(directory) / 'modules/luban0-v1', 'identity')


if __name__ == '__main__':
    unittest.main()
