"""CPU-only source and replay-boundary checks; only local tokenizer is substituted."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_native_decode import IDENTITY, MANIFEST
from capture_q4_decode import replay_record
from q4_sequential_protection import GENERATION, establish_baseline, protection_cases
from q4_task_trajectories import load_task_trajectories


class TaskTrajectoriesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cases = protection_cases()
        self.receipt = {'binary_sha256': 'a'*64}
        self.protocol = {'schema': 'q4-decode-capture-v1', 'vocab_size': 1024, 'eos_ids': [1023],
            'cases': [{'id': f'original-{i}', 'split': 'calibration' if i < 16 else 'observation',
                       'prompt_tokens': [1, 50+i]} for i in range(20)]}
        self.plan = {'schema': 'q4-sequential-plan-v1', 'manifest': MANIFEST, 'core': None, 'eos_ids': [1023],
            'cases': [{'id': c['id'], 'split': c['split'], 'tokens': c['prompt_tokens'],
                       'forced_tokens': [2], 'positions': [0, 1, 2]} for c in self.protocol['cases']]}
        self.prior = {'completed': True, 'mode': 'convert', 'receipt': self.receipt, 'plan': self.plan,
                      'protection_cases': self.cases, 'generation': GENERATION}
        self.loaded = {'base_identity': IDENTITY, 'manifest_sha256': MANIFEST, 'inference_precision': 'bf16',
                       'quantized_projections': 0, 'core_loaded': False, 'core_file_sha256': None}
        self.generations, self.prompts, self.texts = [], {}, {}
        for i, case in enumerate(self.cases):
            prompt, emitted = [1, 100+i], [10+i, 1023]
            check = case['check']
            text = check['expected'] if check['type'] == 'exact' else json.dumps(check['expected'])
            if i == 15:
                text = 'wrong-original-answer'
            self.prompts[json.dumps(case['messages'])] = prompt
            self.texts[tuple(emitted)] = text
            native = {'event': 'result', 'id': 'baseline-'+case['id'], 'tokens': emitted,
                      'complete': True, 'finish_reason': 'stop', 'cached_prompt_tokens': 0,
                      'prompt_tokens': len(prompt), 'generated_tokens': len(emitted)}
            self.generations.append({'id': case['id'], 'request_id': native['id'], 'prompt_tokens': prompt,
                                     'tokens': emitted, 'text': text, 'complete': True, 'native_result': native})
        self.tokenizer = self
        self.tokenizer_patch = patch('q4_task_trajectories.AutoTokenizer.from_pretrained', return_value=self)
        self.loader = self.tokenizer_patch.start()
        self.addCleanup(self.tokenizer_patch.stop)
        self.write()

    def apply_chat_template(self, messages, **kwargs):
        self.assertEqual(kwargs, {'add_generation_prompt': True, 'tokenize': True, 'enable_thinking': False})
        return {'input_ids': self.prompts[json.dumps(messages)]}

    def decode(self, tokens, skip_special_tokens):
        self.assertTrue(skip_special_tokens)
        return self.texts.get(tuple(tokens), 'unrecognized-token-sequence')

    def write(self):
        contents = {'result.json': self.prior, 'loaded.json': self.loaded,
                    'protection-baseline.json': establish_baseline(self.cases, self.generations),
                    'baseline-generations.json': self.generations}
        for name, value in contents.items():
            (self.directory/name).write_text(json.dumps(value))

    def read(self):
        return load_task_trajectories(self.directory, self.receipt, self.plan, self.protocol)

    def test_all_sixteen_include_original_error_and_match_replay_rule(self):
        records = self.read()
        self.assertEqual(len(records), 16)
        self.assertEqual(records[-1]['generated_tokens'], self.generations[-1]['tokens'])
        self.assertEqual([r['id'] for r in records], [c['id'] for c in self.cases])
        self.assertTrue(all(r['split'] == 'calibration' for r in records))
        for row, record in zip(self.generations, records):
            expected = replay_record({'id': row['request_id'], 'split': 'calibration',
                                      'prompt_tokens': row['prompt_tokens']}, row['native_result'], [1023], 1024)
            expected.update(id=row['id'], source_request_id=row['request_id'])
            self.assertEqual(record, expected)
            self.assertEqual(record['assistant_positions'][0], len(row['prompt_tokens']))
            self.assertEqual(record['replay_tokens'][-1], row['tokens'][0])
            self.assertNotIn(1023, record['replay_tokens'])
        self.loader.assert_called_once()

    def test_eos_first_has_no_assistant_input_and_length_budget_retained(self):
        first, second = self.generations[:2]
        first.update(tokens=[1023], text='')
        first['native_result'].update(tokens=[1023], generated_tokens=1)
        self.texts[(1023,)] = ''
        emitted = [11]*GENERATION['max_tokens']
        second.update(tokens=emitted, complete=False)
        second['native_result'].update(tokens=emitted, complete=False, finish_reason='length', generated_tokens=len(emitted))
        self.texts[tuple(emitted)] = second['text']
        self.write()
        records = self.read()
        self.assertEqual(records[0]['assistant_positions'], [])
        self.assertEqual(records[0]['replay_tokens'], first['prompt_tokens'])
        self.assertFalse(records[1]['complete'])
        self.assertEqual(len(records[1]['replay_tokens']), len(second['prompt_tokens'])+191)
        self.assertEqual(records[1]['assistant_positions'][-1], len(second['prompt_tokens'])+190)

    def test_changed_prompt_decoded_text_and_candidate_substitution_rejected(self):
        original = deepcopy(self.generations)
        for change in ('prompt', 'decoded-text', 'candidate-id', 'raw-tokens', 'raw-complete', 'cache', 'prompt-count', 'output-count'):
            with self.subTest(change=change):
                self.generations = deepcopy(original)
                row = self.generations[0]; raw = row['native_result']
                if change == 'prompt': row['prompt_tokens'][0] = 8
                elif change == 'decoded-text': row['text'] = 'forged'
                elif change == 'candidate-id': row['request_id'] = raw['id'] = 'fixed-final-'+row['id']
                elif change == 'raw-tokens': raw['tokens'] = [15, 1023]
                elif change == 'raw-complete': raw['complete'] = False
                elif change == 'cache': raw['cached_prompt_tokens'] = 1
                elif change == 'prompt-count': raw['prompt_tokens'] += 1
                else: raw['generated_tokens'] += 1
                self.write()
                with self.assertRaises(ValueError): self.read()

    def test_missing_duplicate_and_forged_baseline_scores_rejected(self):
        for change in ('missing', 'duplicate', 'score', 'different-baseline-file'):
            with self.subTest(change=change):
                self.write()
                path = self.directory/'baseline-generations.json'
                rows = deepcopy(self.generations)
                if change == 'missing': rows.pop()
                elif change == 'duplicate': rows[-1] = rows[0]
                elif change == 'different-baseline-file': rows[0]['text'] = 'candidate-answer'
                else:
                    baseline = json.loads((self.directory/'protection-baseline.json').read_text())
                    baseline['correct_ids'].append(self.cases[-1]['id'])
                    (self.directory/'protection-baseline.json').write_text(json.dumps(baseline))
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError): self.read()

    def test_eos_token_and_context_boundaries_rejected(self):
        original = deepcopy(self.generations)
        for tokens in ([1023, 10, 1023], [1024, 1023], [True, 1023], [10], [10]*193):
            with self.subTest(tokens=tokens[:3]):
                self.generations = deepcopy(original)
                row = self.generations[0]
                row.update(tokens=tokens, complete=tokens[-1] == 1023)
                row['native_result'].update(tokens=tokens, generated_tokens=len(tokens), complete=row['complete'],
                                             finish_reason='stop' if row['complete'] else 'length')
                if any(type(t) is not int for t in tokens) or len(tokens)>192:
                    (self.directory/'baseline-generations.json').write_text(json.dumps(self.generations))
                else: self.write()
                with self.assertRaises(ValueError): self.read()
        self.generations = deepcopy(original)
        self.generations[0]['prompt_tokens'] = [1]*2048
        self.generations[0]['native_result']['prompt_tokens'] = 2048
        self.write()
        with self.assertRaises(ValueError): self.read()

    def test_source_completion_receipt_plan_cases_and_loaded_binding(self):
        original = deepcopy(self.prior)
        for field, value in [('completed', False), ('receipt', {}), ('plan', {}), ('protection_cases', []), ('generation', {})]:
            with self.subTest(field=field):
                self.prior = {**original, field: value}; self.write()
                with self.assertRaises(ValueError): self.read()
        self.prior = original
        self.loaded['quantized_projections'] = 32
        self.write()
        with self.assertRaises(ValueError): self.read()


if __name__ == '__main__':
    unittest.main()
