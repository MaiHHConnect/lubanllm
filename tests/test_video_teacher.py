"""Joint video teacher requests, source split/review, and real AVFoundation frames."""
import base64
import copy
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image
import test_media_teacher
from luban.teacher import Teacher
from luban.video_learning import extract_frames


class VideoTeacherTests(unittest.TestCase):
    def setUp(self):
        test_media_teacher.MediaTeacherTests.setUp(self)
        self.settings.update({'teacher': {'teaching_modalities': ['video']}})

    def body(self):
        body = test_media_teacher.MediaTeacherTests.body(self)
        body.update(kind='video', teacher_modality='video')
        for i, row in enumerate(body['files']):
            row.update(name=f'{i}.mp4', content=base64.b64encode(b'video fixture' + bytes([i])).decode())
        return body

    @staticmethod
    def frames(source, directory):
        directory.mkdir(mode=0o700)
        result = []
        for i in range(2):
            path = directory / f'{i}.jpg'
            Image.new('RGB', (4, 4), (i * 50, 10, 20)).save(path)
            path.chmod(0o600)
            result.append({'path': str(path), 'timestamp_seconds': i + 0.25})
        return result

    def assert_joint_request(self, payload, saved):
        parts = payload['messages'][0]['content']
        images = [(index, part) for index, part in enumerate(parts) if part['type'] == 'image_url']
        self.assertEqual(len(images), len(saved['video_frames']))
        for (index, part), frame in zip(images, saved['video_frames']):
            self.assertGreater(index, 0)
            self.assertEqual(parts[index - 1]['type'], 'text')
            # Actual time is written before its corresponding frame, in order.
            stamp = float(frame['timestamp_seconds'])
            text = parts[index - 1]['text']
            self.assertTrue(any(abs(float(value) - stamp) < 0.0001
                                for value in re.findall(r'\d+(?:\.\d+)?', text)), text)
            self.assertEqual(part['image_url']['url'], 'data:image/jpeg;base64,' +
                             base64.b64encode(Path(frame['asset_path']).read_bytes()).decode())
        text = '\n'.join(part['text'] for part in parts if part['type'] == 'text')
        self.assertIn(saved['question'], text)
        self.assertIn('音轨', text)
        self.assertNotIn('asset_path', json.dumps(payload))
        self.assertNotIn(str(self.runtime.memory.root), json.dumps(payload))

    def test_video_only_scope_one_joint_qa_per_clip_and_review_before_training(self):
        self.assertFalse(self.media.status('image')['teacher_enabled'])
        self.assertTrue(self.media.status('video')['teacher_enabled'])
        self.assertEqual(self.media.status('video')['capability']['required_available_bytes'], 10_352_340_672)
        with patch('luban.video_learning.extract_frames', self.frames):
            public = self.media.create(self.body())
        identifier = public['id']
        self.assertEqual((public['kind'], public['teacher_modality'], public['video_count'], public['frame_count']),
                         ('video', 'video', 2, 4))
        rows = self.media._load(identifier)['samples']
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['split'] for r in rows], ['training', 'validation'])
        self.assertNotEqual(rows[0]['family'], rows[1]['family'])
        self.assertTrue(all(r['kind'] == 'video' and len(r['video_frames']) == 2 for r in rows))
        with self.assertRaises(ValueError): self.media.start(identifier)
        self.practice.run_one()
        self.assertEqual(self.media._load(identifier)['state'], 'teacher_queued')
        self.practice.run_one()
        body = self.media._load(identifier)
        self.assertEqual(body['state'], 'teacher_review')
        self.assertEqual((len(self.sent), self.teacher.calls), (2, 2))
        for payload, row in zip(self.sent, rows): self.assert_joint_request(payload, row)
        public_json = json.dumps(self.media.public(body))
        self.assertNotIn('asset_path', public_json); self.assertNotIn(str(self.runtime.memory.root), public_json)
        self.assertNotIn('base64', public_json)
        self.assertTrue(all(not row['approved'] for row in body['samples']))
        with self.assertRaises(ValueError): self.media.start(identifier)
        partial = [{k: row[k] for k in ('id', 'question', 'answer')} for row in body['samples'][:1]]
        with self.assertRaises(ValueError): self.media.confirm_teacher({'id': identifier, 'answers_reviewed': True, 'samples': partial})
        value = self.media.confirm_teacher({'id': identifier, 'answers_reviewed': True,
            'samples': [{k: row[k] for k in ('id', 'question', 'answer')} for row in body['samples']]})
        self.assertEqual(value['state'], 'ready')
        self.assertTrue(all(row['kind'] == 'video' and row['approved'] and len(row['video_frames']) == 2
                            for row in self.media._load(identifier)['samples']))
        self.assertFalse(value['adopted'])

    def test_disabled_limits_duplicate_no_consent_and_legacy_request_refused_before_decode(self):
        bad = []
        body = self.body(); body['files'] *= 3; bad.append(body)
        body = self.body(); body['teacher_confirmed'] = False; bad.append(body)
        body = self.body(); body['files'][1]['content'] = body['files'][0]['content']; bad.append(body)
        body = self.body(); body['kind'] = 'image'; bad.append(body)
        with patch('luban.video_learning.extract_frames') as decode:
            for value in bad:
                with self.assertRaises(ValueError): self.media.create(value)
            self.settings.update({'teacher': {'teaching_modalities': ['image', 'audio', 'text']}})
            with self.assertRaises(ValueError): self.media.create(self.body())
            decode.assert_not_called()
        self.assertEqual(self.sent, [])

    def test_decode_failure_never_queues_or_spends(self):
        with patch('luban.video_learning.extract_frames', side_effect=ValueError('invalid clip')):
            with self.assertRaises(ValueError): self.media.create(self.body())
        self.assertIsNone(self.media.teacher_pending()); self.assertEqual(self.teacher.calls, 0)

    def test_joint_request_durable_before_payment_failure_not_retried(self):
        with patch('luban.video_learning.extract_frames', self.frames):
            identifier = self.media.create(self.body())['id']
        requests = []
        def respond(request):
            self.assertEqual(self.media._load(identifier)['state'], 'teacher_running')
            requests.append(json.loads(request.content))
            return httpx.Response(429, text='private response')
        self.teacher.transport = httpx.MockTransport(respond)
        self.practice.run_one()
        saved = self.media._load(identifier)
        self.assertEqual(saved['state'], 'teacher_failed')
        self.assertEqual((len(requests), self.teacher.calls), (1, 1))
        self.assert_joint_request(requests[0], saved['samples'][0])
        self.assertTrue(all(not row['approved'] for row in saved['samples']))
        self.practice.run_one()
        self.assertEqual(len(requests), 1)
        self.assertNotIn('private response', saved['error'])

    def test_old_per_frame_image_history_is_not_relabelled_joint_video(self):
        with patch('luban.video_learning.extract_frames', self.frames):
            identifier = self.media.create(self.body())['id']
        legacy = self.media._load(identifier)
        legacy.pop('learning_mode', None)
        legacy.update(id='a' * 32, kind='image', teacher_modality='video', state='awaiting_quality', candidate='b' * 12)
        frame = legacy['samples'][0]['video_frames'][0]
        legacy['samples'] = [{'id': 'legacy-frame', 'name': 'frame.jpg', 'kind': 'image',
            'question': '单帧是什么？', 'answer': '原单帧标注', 'family': 'old-source', 'split': 'training',
            'asset_path': frame['asset_path'], 'timestamp_seconds': frame['timestamp_seconds']}]
        before = copy.deepcopy(legacy)
        self.media._write(legacy)
        self.assertEqual(self.media._load(legacy['id']), before)
        self.assertEqual(self.media.status('video')['lesson']['id'], identifier)
        self.assertEqual(self.media.status('image')['lesson']['kind'], 'image')
        self.assertNotIn('video_frames', self.media._load(legacy['id'])['samples'][0])

    def test_real_native_two_clips_joint_teacher_review_without_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = root / 'fixture'
            build = subprocess.run(['/usr/bin/xcrun', 'swiftc', 'tests/fixtures/video_frames.swift',
                '-O', '-framework', 'AVFoundation', '-o', str(fixture)], capture_output=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr.decode())
            body = self.body()
            for i, row in enumerate(body['files']):
                source = root / f'{i}.mov'
                result = subprocess.run([str(fixture), str(source), str(i + 2)], capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                row.update(name=source.name, content=base64.b64encode(source.read_bytes()).decode())
            value = self.media.create(body)
            self.assertEqual((value['video_count'], value['frame_count']), (2, 8))
            self.practice.run_one(); self.practice.run_one()
            saved = self.media._load(value['id'])
            self.assertEqual(saved['state'], 'teacher_review')
            self.assertEqual((len(self.sent), self.teacher.calls), (2, 2))
            self.assertEqual([r['split'] for r in saved['samples']], ['training', 'validation'])
            for payload, row in zip(self.sent, saved['samples']):
                stamps = [frame['timestamp_seconds'] for frame in row['video_frames']]
                self.assertTrue(all(stamp >= 0 for stamp in stamps)); self.assertEqual(stamps, sorted(set(stamps)))
                self.assert_joint_request(payload, row)
            self.assertTrue(all(not r['approved'] for r in saved['samples']))
            self.assertEqual(self.runtime.state, 'stopped')

    def test_teacher_standard_interleaved_wire_sequence_is_preserved(self):
        payloads = []
        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': '动作依次发生'}, 'finish_reason': 'stop'}]})
        teacher = Teacher(self.settings, httpx.MockTransport(respond))
        parts = [{'type': 'text', 'text': '按真实时间顺序联合观察画面，未提供音轨。'}]
        for index, stamp in enumerate((0.125, 1.5, 2.25, 4.0)):
            parts.extend([{'type': 'text', 'text': f'时间戳 {stamp} 秒'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' +
                    base64.b64encode(b'frame' + bytes([index])).decode()}}])
        messages = [{'role': 'user', 'content': parts}]
        original = copy.deepcopy(messages)
        list(teacher.stream(messages, 100, modality='video'))
        self.assertEqual((len(payloads), teacher.calls), (1, 1))
        self.assertEqual(payloads[0]['messages'], original); self.assertEqual(messages, original)

    def test_bad_decode_manifest_or_escaped_frame_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for index, body in enumerate(({'duration_seconds': 31, 'frames': []},
                    {'duration_seconds': 2, 'frames': [{'path': str(root / str(1) / 'frame.jpg'), 'timestamp_seconds': -.5}]},
                    {'duration_seconds': 2, 'frames': [{'path': str(root / 'outside.jpg'), 'timestamp_seconds': .5}]})):
                with patch('luban.video_learning.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(body).encode())):
                    with self.assertRaises(ValueError): extract_frames(root / 'source.mp4', root / str(index))


if __name__ == '__main__': unittest.main()
