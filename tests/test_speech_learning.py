import json
from pathlib import Path
import struct
import sys
import threading
import wave

import tempfile
import unittest

from luban.speech_learning import (_binary, _descriptor, _filters, _memory_required,
                                   _prepare, _read, _recognition_quality, _round_progress, _run, validate_samples)
from luban.training_checkpoint import TrainingPaused


def sample(tmp_path, name='train', split='training', family='speaker-one'):
    return dict(id=name, kind='sound', split=split, family=family, answer='你好。',
                asset_path=str(tmp_path / (name + '.wav')), training_allowed=True, transcript_reviewed=True)


class SpeechLearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_partial_round_only_resumes_remaining_budget(self):
        self.assertEqual(_round_progress(None), (0, 0, 4, 0))
        row = dict(state='paused', total_adam_step=6, completed_steps=2, round_budget=4)
        self.assertEqual(_round_progress(row), (6, 2, 2, 4))
        self.assertEqual(_round_progress({**row, 'state': 'awaiting_quality', 'completed_steps': 4,
                                         'total_adam_step': 8}), (8, 0, 4, 8))

    def test_inconsistent_saved_budget_or_steps_rejected(self):
        row = dict(state='paused', total_adam_step=2, completed_steps=2, round_budget=4)
        for changed in ({'completed_steps': 5}, {'total_adam_step': 1}, {'round_budget': 8}, {'completed_steps': True}):
            with self.assertRaises(ValueError):
                _round_progress({**row, **changed})

    def test_permissions_split_and_complete_labels(self):
        tmp_path = self.directory
        a, b = sample(tmp_path), sample(tmp_path, 'held', 'validation', 'speaker-two')
        assert validate_samples([a, b])[0]['answer'] == '你好。'
        for changed in ({'training_allowed': False}, {'transcript_reviewed': False}, {'answer': 'a<|im_end|>'}):
            with self.assertRaises(ValueError):
                validate_samples([{**a, **changed}, b])
        with self.assertRaisesRegex(ValueError, '同一声音来源'):
            validate_samples([a, {**b, 'family': a['family']}])
        with self.assertRaises(ValueError):
            validate_samples([a])


    def test_symlink_input_rejected(self):
        tmp_path = self.directory
        source, link = tmp_path / 'wav', tmp_path / 'link'
        source.write_bytes(b'complete')
        link.symlink_to(source)
        with self.assertRaises(OSError):
            _read(link, 20)
        with self.assertRaises(ValueError):
            _read(source, 2)


    def test_process_array_protocol_and_normal_close(self):
        tmp_path = self.directory
        result, receipt = _run([sys.executable, '-c', 'print("[1,2]")'], tmp_path, None)
        assert result == [1, 2] and receipt['exit_code'] == 0


    def test_process_cancel_waits_for_actual_exit(self):
        tmp_path = self.directory
        event = threading.Event()
        timer = threading.Timer(.1, event.set)
        timer.start()
        try:
            with self.assertRaises(TrainingPaused):
                _run([sys.executable, '-c', 'import time; time.sleep(10)'], tmp_path, event)
        finally:
            timer.cancel()
        receipt = json.loads((tmp_path / 'process.json').read_text())
        assert receipt['exit_code'] == -15


    def test_complete_native_checkpoint_event_is_delivered(self):
        events = []
        script = 'import json; print(json.dumps({"event":"checkpoint","step":2,"sample":1})); print(json.dumps({"passed":True}))'
        result, receipt = _run([sys.executable, '-c', script], self.directory, None, checkpoint=events.append)
        self.assertEqual(events, [dict(event='checkpoint', step=2, sample=1)])
        self.assertTrue(result['passed'])
        self.assertEqual(receipt['exit_code'], 0)


    def test_checkpoint_publication_failure_closes_child_and_records_wait(self):
        def rejected(_):
            raise ValueError('publication refused')
        script = 'import json,time; print(json.dumps({"event":"checkpoint","step":1,"sample":0}),flush=True); time.sleep(10)'
        with self.assertRaisesRegex(ValueError, 'publication refused'):
            _run([sys.executable, '-c', script], self.directory, None, checkpoint=rejected)
        receipt = json.loads((self.directory / 'process.json').read_text())
        self.assertEqual(receipt['exit_code'], -15)


    def test_memory_grows_with_complete_input(self):
        inputs = lambda tokens, frames: [dict(tokens=tokens, padded_frames=frames, chunks=[('input', 'f32', [1], b'1234')])]
        assert _memory_required(782426112, 311164928, inputs(89, 500)) > _memory_required(782426112, 311164928, inputs(29, 100))


    def test_native_preparation_uses_real_tokenizer_and_complete_eos(self):
        tmp_path = self.directory
        from luban.speech import DEFAULT_SPEECH_MODEL_PATH
        model = Path(DEFAULT_SPEECH_MODEL_PATH)
        try:
            features, tokenizer = _binary('luban-asr-features'), _binary('luban-asr-tokenizer')
        except RuntimeError:
            self.skipTest('Native DSP/tokenizer not built')
        if not model.exists():
            self.skipTest('Local tokenizer metadata not installed')
        wav = tmp_path / 'authored.wav'
        with wave.open(str(wav), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            out.writeframes(struct.pack('<8000h', *([0] * 8000)))
        output = tmp_path / 'prepared'
        output.mkdir()
        filters = tmp_path / 'filters.raw'
        filters.write_bytes(_filters())
        with filters.open('rb') as fd:
            result = _prepare(sample(tmp_path), wav.read_bytes(), output, fd.fileno(), model,
                              {'features': features, 'tokenizer': tokenizer}, None)
        ids = struct.unpack('<' + 'q' * result['tokens'], result['chunks'][2][3])
        labels = struct.unpack('<' + 'q' * result['tokens'], result['chunks'][3][3])
        assert ids[-1] == labels[-1] == 151645
        assert result['padded_frames'] == 100 and labels.count(-100) > 0
        prompt = result['prompt_chunks']
        self.assertEqual([c[0] for c in prompt], ['input', 'mask', 'ids'])
        self.assertEqual(struct.unpack('<' + 'q' * result['prompt_tokens'], prompt[2][3]), ids[:labels.count(-100)])
        self.assertNotIn(151645, struct.unpack('<' + 'q' * result['prompt_tokens'], prompt[2][3])[-1:])

    def test_free_generation_metrics_keep_errors_and_require_eos(self):
        rows = [sample(self.directory)]
        def decoded(text, ended=True):
            return [dict(ok=True, decode=text, ids=[1, 151645] if ended else [1] * 512)]
        quality = _recognition_quality(rows, decoded('language Chinese<asr_text>你好。<|im_end|>'))
        self.assertTrue(quality['passed'])
        self.assertEqual((quality['CER'], quality['WER']), (0, 0))
        self.assertFalse(_recognition_quality(rows, decoded('你好。', False))['passed'])
        repeat = _recognition_quality(rows, decoded('<asr_text>你好你好<|im_end|>'))
        self.assertEqual(repeat['character_edits'], 2)
        self.assertEqual(repeat['CER'], 1)
        self.assertFalse(repeat['passed'])
        self.assertFalse(_recognition_quality(rows, decoded('不认识<|im_end|>'))['passed'])
        english = [{**rows[0], 'answer': 'Hello world'}]
        exact = _recognition_quality(english, decoded('ＨＥＬＬＯ, WORLD!<|im_end|>'))
        self.assertTrue(exact['passed'])
        inserted = _recognition_quality(english, decoded('hello brave world<|im_end|>'))
        self.assertEqual(inserted['WER'], .5)
        self.assertEqual(inserted['CER'], .5)
        with self.assertRaises(ValueError):
            _recognition_quality(rows, [])

    def test_holdout_cache_budget_increases_memory_without_changing_training_logits(self):
        item = dict(tokens=89, prompt_tokens=65, padded_frames=500, chunks=[], record={'split': 'training'})
        training = _memory_required(782426112, 311164928, [item])
        validation = _memory_required(782426112, 311164928, [{**item, 'record': {'split': 'validation'}}])
        self.assertEqual(validation - training, 2 * 28 * 8 * 128 * (65 + 512 - 89) * 4)

