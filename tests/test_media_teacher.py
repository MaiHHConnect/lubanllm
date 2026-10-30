"""Private SQLite + real compatible mock HTTP; no paid API or model loads."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import httpx
from PIL import Image

from luban.automatic import AutomaticLearning
from luban.media_learning import MediaLearning
from luban.memory import Memory
from luban.settings import Settings
from luban.teacher import Teacher
from luban.teacher_practice import TeacherPractice


class MediaTeacherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        self.settings = Settings(root / 'home')
        self.settings.update({'storage_root': str(root / 'storage'), 'teacher': {
            'base_url': 'http://offline.invalid/v1', 'api_key': 'fixture-secret', 'model': 'fixture',
            'teaching_modalities': ['image', 'audio']}})
        self.settings.accept_agreement()
        self.runtime = SimpleNamespace(memory=Memory(root / 'storage'), state='stopped',
            job=None, lock=threading.RLock(), active_core=None, identity=None, store=None)
        self.sent = []
        def respond(request):
            self.sent.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': '鲁班。'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 8, 'completion_tokens': 2, 'total_tokens': 10}})
        self.teacher = Teacher(self.settings, httpx.MockTransport(respond))
        self.media = MediaLearning(self.runtime, self.settings, self.teacher)
        self.media._kind = lambda _: SimpleNamespace(ROUND_STEPS=5, capability=lambda _: {'available': False,
            'required_available_bytes': 7_968_468_672})
        self.practice = TeacherPractice(self.runtime, self.settings, self.teacher)
        self.practice.media_learning = self.media
        self.stopped = self.practice.stop_event

    def body(self, kind='image'):
        files = []
        for i in range(2):
            stream = io.BytesIO()
            if kind == 'image':
                Image.new('RGB', (2, 2), (i, 30, 60)).save(stream, format='PNG')
            else:
                with wave.open(stream, 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                    wav.writeframes(bytes([i + 1, 0]) * 160)
            files.append({'name': f'{i}.png' if kind == 'image' else f'{i}.wav', 'question': '是谁？',
                'answer': '', 'family': str(i), 'content': base64.b64encode(stream.getvalue()).decode()})
        return {'kind': kind, 'teacher_requested': True, 'teacher_confirmed': True,
                'training_allowed': True, 'answers_reviewed': False, 'files': files}

    def finish(self, identifier):
        self.practice.run_one(); self.practice.run_one()
        return self.media._load(identifier)

    def test_real_image_parts_idle_queue_and_explicit_review_before_training(self):
        value = self.media.create(self.body()); identifier = value['id']
        self.assertEqual(value['state'], 'teacher_queued'); self.assertEqual(self.sent, [])
        self.assertTrue(self.practice.pending())  # no text model or identity needed
        with self.assertRaises(ValueError): self.media.start(identifier)
        body = self.finish(identifier)
        self.assertEqual(body['state'], 'teacher_review'); self.assertEqual(len(self.sent), 2)
        self.assertTrue(self.sent[0]['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertTrue(all(not r['approved'] and not r['answers_reviewed'] and not r['training_allowed'] for r in body['samples']))
        self.assertEqual(self.runtime.memory.usage()['teacher']['total_tokens'], 20)
        public = self.media.public(body)
        self.assertNotIn('asset_path', json.dumps(public)); self.assertNotIn('base64', json.dumps(public))
        with self.assertRaises(ValueError): self.media.confirm_teacher({'id': identifier, 'answers_reviewed': False})
        rows = [{k: r[k] for k in ('id', 'question', 'answer')} for r in body['samples']]
        rows[0]['answer'] = '用户核对的鲁班'
        value = self.media.confirm_teacher({'id': identifier, 'samples': rows, 'answers_reviewed': True})
        self.assertEqual(value['state'], 'ready'); self.assertFalse(value['adopted'])
        self.assertTrue(all(r['approved'] for r in self.media._load(identifier)['samples']))

    def test_actual_audio_request_preserves_wav_and_no_teacher_truth_claim(self):
        source = self.body('sound'); identifier = self.media.create(source)['id']
        saved = self.finish(identifier)
        self.assertEqual(saved['state'], 'teacher_review')
        part = self.sent[0]['messages'][0]['content'][1]
        self.assertEqual(part['input_audio'], {'data': source['files'][0]['content'], 'format': 'wav'})
        self.assertTrue(all(not r['transcript_reviewed'] for r in saved['samples']))

    def test_disabled_no_consent_invalid_asset_and_configuration_change_never_call(self):
        body = self.body(); body['teacher_confirmed'] = False
        with self.assertRaises(ValueError): self.media.create(body)
        identifier = self.media.create(self.body())['id']
        self.settings.update({'teacher': {'teaching_modalities': []}})
        self.assertFalse(self.practice.pending()); self.practice.run_one(); self.assertEqual(self.sent, [])
        self.settings.update({'teacher': {'teaching_modalities': ['image'], 'model': 'different'}})
        self.practice.run_one()
        self.assertEqual(self.media._load(identifier)['state'], 'teacher_failed'); self.assertEqual(self.sent, [])
        body = self.body(); body['files'][0]['content'] = base64.b64encode(b'not an image').decode()
        other = self.media.create(body)['id']; self.practice.run_one()
        self.assertEqual(self.media._load(other)['state'], 'teacher_failed'); self.assertEqual(self.sent, [])

    def test_interrupted_paid_attempt_durable_no_automatic_retry(self):
        identifier = self.media.create(self.body())['id']
        def failed(*args, **kwargs):
            self.assertEqual(self.media._load(identifier)['state'], 'teacher_running')
            self.teacher.calls += 1
            raise RuntimeError('private provider detail')
            yield
        with patch.object(self.teacher, 'stream', failed): self.practice.run_one()
        body = self.media._load(identifier)
        self.assertEqual(body['state'], 'teacher_failed'); self.assertNotIn('private provider', body['error'])
        self.assertFalse(self.practice.pending()); self.practice.run_one(); self.assertEqual(self.sent, [])
        self.assertIsNone(self.runtime.memory.usage()['teacher']['total_tokens'])

    def test_restart_marks_unknown_request_interrupted_without_retry(self):
        identifier = self.media.create(self.body())['id']; body = self.media._load(identifier)
        body.update(state='teacher_running', owner={'pid': 99999999, 'started': 1})
        self.media._write(body)
        self.assertFalse(self.practice.pending())
        self.assertEqual(self.media._load(identifier)['state'], 'teacher_interrupted'); self.assertEqual(self.sent, [])

    def test_private_asset_scope_and_below_root_symlinks_reject_before_network(self):
        for mode in ('outside', 'parent-link'):
            identifier = self.media.create(self.body())['id']; body = self.media._load(identifier)
            path = Path(body['samples'][0]['asset_path'])
            if mode == 'outside':
                outside = Path(self.settings.home) / 'not-a-lesson.png'
                outside.write_bytes(path.read_bytes())
                body['samples'][0]['asset_path'] = str(outside); self.media._write(body)
            else:
                moved = path.parent.with_name(path.parent.name + '-retained')
                path.parent.rename(moved); path.parent.symlink_to(moved, target_is_directory=True)
            self.practice.run_one()
            self.assertEqual(self.media._load(identifier)['state'], 'teacher_failed')
            self.assertEqual(self.sent, [])

    def test_automatic_idle_reuses_command_lock_without_loading_local_model(self):
        self.media.create(self.body())
        lock = threading.Lock()
        automatic = AutomaticLearning(self.runtime, self.settings, lock, lambda: False)
        automatic.teacher_practice = self.practice; automatic.last_activity = 0
        automatic.tick(); automatic.worker.join(3)
        self.assertFalse(automatic.worker.is_alive()); self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.runtime.state, 'stopped'); self.assertTrue(lock.acquire(False)); lock.release()
        automatic.last_activity = 0
        automatic.tick(); automatic.worker.join(3)
        self.assertEqual(len(self.sent), 2); self.assertFalse(self.media.teacher_pending())


if __name__ == '__main__': unittest.main()
