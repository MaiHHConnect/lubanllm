"""Protocol validation uses mock HTTP only; no paid teacher or model execution."""
import base64
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx

from luban.settings import Settings
from luban.teacher import Teacher


class TeacherModalityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.settings = Settings(tmp.name)
        self.settings.update({'teacher': {'base_url': 'https://offline.invalid/v1',
            'api_key': 'private-key', 'model': 'mock', 'session_call_limit': 20}})
        self.sent = []
        def respond(request):
            self.sent.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': '答'}, 'finish_reason': 'stop'}]})
        self.teacher = Teacher(self.settings, httpx.MockTransport(respond))

    @staticmethod
    def messages(kind='text', raw=b'private-fixture'):
        part = {'type': 'text', 'text': '请核对'}
        content = [part]
        encoded = base64.b64encode(raw).decode('ascii')
        if kind == 'image':
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}})
        elif kind == 'audio':
            content.append({'type': 'input_audio', 'input_audio': {'data': encoded, 'format': 'wav'}})
        return [{'role': 'user', 'content': content}]

    def reject(self, messages, modality='text'):
        before = len(self.sent), self.teacher.calls
        with self.assertRaises(ValueError) as error:
            list(self.teacher.stream(messages, 10, modality=modality))
        self.assertNotIn('private-fixture', str(error.exception))
        self.assertNotIn(base64.b64encode(b'private-fixture').decode(), str(error.exception))
        self.assertEqual((len(self.sent), self.teacher.calls), before)

    def test_default_text_compatibility_and_missing_setting_migration(self):
        self.assertEqual(list(self.teacher.stream([{'role': 'user', 'content': '你好'}], 10)),
                         [{'content': '答'}, {'finish_reason': 'stop'}])
        snapshot = self.settings.snapshot(); del snapshot['teacher']['teaching_modalities']
        with patch.object(self.settings, 'snapshot', return_value=snapshot):
            list(self.teacher.stream(self.messages(), 10))
        self.assertEqual(self.teacher.calls, 2)

    def test_disabled_each_modality_does_not_send_or_reserve_calls(self):
        for enabled in ([], ['text'], ['image'], ['audio']):
            self.settings.update({'teacher': {'teaching_modalities': enabled}})
            for modality in {'text', 'image', 'audio'} - set(enabled):
                self.reject(self.messages(modality), modality)

    def test_standard_image_and_audio_payloads_pass_unchanged_without_text_permission(self):
        self.settings.update({'teacher': {'teaching_modalities': ['image', 'audio']}})
        for modality in ('image', 'audio'):
            messages = self.messages(modality)
            original = copy.deepcopy(messages)
            list(self.teacher.stream(messages, 20, modality=modality))
            self.assertEqual(self.sent[-1]['messages'], original)
            self.assertEqual(messages, original)
        self.assertEqual(self.teacher.calls, 2)

    def test_declared_text_and_cross_modality_cannot_bypass_permission(self):
        self.settings.update({'teacher': {'teaching_modalities': ['text', 'image', 'audio']}})
        for actual, declared in (('image', 'text'), ('audio', 'text'), ('image', 'audio'), ('audio', 'image'), ('text', 'image')):
            self.reject(self.messages(actual), declared)
        mixed = self.messages('image'); mixed[0]['content'] += self.messages('audio')[0]['content']
        self.reject(mixed, 'image')

    def test_raw_video_and_hidden_attachment_paths_are_rejected(self):
        self.settings.update({'teacher': {'teaching_modalities': ['text', 'image', 'video', 'audio']}})
        self.reject(self.messages(), 'video')
        for kind in ('video', 'video_url', 'input_video', 'file', 'input_file', 'input_image'):
            self.reject([{'role': 'user', 'content': [{'type': kind, kind: 'private-fixture'}]}])
        for field in ('attachments', 'image', 'audio', 'input_audio', 'video_url'):
            self.reject([{'role': 'user', 'content': 'text', field: 'private-fixture'}])
        self.reject(self.messages(), [])

    def video_messages(self, count=4, raw=b'private-fixture'):
        content = [{'type': 'text', 'text': '以下为视频抽取的画面帧，仅画面，不含音轨；请勿推测声音。'}]
        for index in range(count):
            content.append({'type': 'text', 'text': f'时间戳 {index * 0.5:.3f} 秒'})
            content.append(self.messages('image', raw)[0]['content'][1])
        return [{'role': 'user', 'content': content}]

    def test_video_only_permission_accepts_timestamped_frames_unchanged(self):
        self.settings.update({'teacher': {'teaching_modalities': ['video']}})
        for count in (1, 4):
            messages = self.video_messages(count)
            original = copy.deepcopy(messages)
            list(self.teacher.stream(messages, 20, modality='video'))
            self.assertEqual(self.sent[-1]['messages'], original)
            self.assertEqual(messages, original)
        self.assertEqual(self.teacher.calls, 2)
        self.reject(self.messages())
        self.reject(self.messages('image'), 'image')
        self.reject(self.messages('audio'), 'audio')

    def test_video_closed_is_not_enabled_by_image_or_text(self):
        self.settings.update({'teacher': {'teaching_modalities': ['text', 'image', 'audio']}})
        self.reject(self.video_messages(), 'video')
        self.reject(self.video_messages(), 'text')

    def test_video_rejects_no_frames_excess_frames_audio_and_raw_video(self):
        self.settings.update({'teacher': {'teaching_modalities': ['video', 'audio']}})
        self.reject(self.video_messages(0), 'video')
        self.reject(self.video_messages(5), 'video')
        mixed = self.video_messages(1); mixed[0]['content'].append(self.messages('audio')[0]['content'][1])
        self.reject(mixed, 'video')
        for kind in ('video_url', 'input_video'):
            self.reject([{'role': 'user', 'content': [{'type': kind, kind: 'https://private.invalid/video.mp4'}]}], 'video')

    def test_video_reuses_aggregate_size_and_local_frame_validation(self):
        self.settings.update({'teacher': {'teaching_modalities': ['video']}})
        with patch('luban.teacher.MAX_MEDIA_BYTES', 8):
            self.reject(self.video_messages(1, b'123456789'), 'video')
            self.reject(self.video_messages(2, b'12345'), 'video')
        messages = self.video_messages(1)
        messages[0]['content'][-1]['image_url']['url'] = 'https://private.invalid/frame.png'
        self.reject(messages, 'video')

    def test_strict_media_schema_bad_base64_and_remote_urls(self):
        self.settings.update({'teacher': {'teaching_modalities': ['text', 'image', 'audio']}})
        bad_parts = [
            {'type': 'image_url', 'image_url': {'url': 'https://private.invalid/photo.png'}},
            {'type': 'image_url', 'image_url': {'url': 'data:image/svg+xml;base64,AAAA'}},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,A'}},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,', 'detail': []}},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,', 'extra': 'private-fixture'}},
            {'type': 'input_audio', 'input_audio': {'data': '!private-fixture!', 'format': 'wav'}},
            {'type': 'input_audio', 'input_audio': {'data': 'AAAA', 'format': 'flac'}},
            {'type': 'input_audio', 'input_audio': {'data': '', 'format': 'wav'}},
            {'type': 'input_audio', 'input_audio': {'data': 'AAAA', 'format': []}},
            {'type': [], 'text': 'private-fixture'},
            {'type': 'text', 'text': 'okay', 'input_audio': {'data': 'private-fixture'}}]
        for part in bad_parts:
            with self.subTest(part_type=part.get('type')):
                self.reject([{'role': 'user', 'content': [part]}],
                            'audio' if part.get('type') == 'input_audio' else 'image')
        system = self.messages('image'); system[0]['role'] = 'system'
        self.reject(system, 'image')

    def test_media_count_and_decoded_aggregate_are_bounded_before_quota(self):
        self.settings.update({'teacher': {'teaching_modalities': ['image', 'audio']}})
        oversized = self.messages('image', b'123456789')
        with patch('luban.teacher.MAX_MEDIA_BYTES', 8):
            self.reject(oversized, 'image')
            combined = self.messages('image', b'12345')
            combined[0]['content'] += self.messages('image', b'12345')[0]['content']
            self.reject(combined, 'image')
        many = self.messages('audio')
        many[0]['content'] = [many[0]['content'][1]] * 5
        self.reject(many, 'audio')

    def test_network_failure_counts_one_attempt_and_does_not_retry(self):
        self.settings.update({'teacher': {'teaching_modalities': ['image'], 'session_call_limit': 1}})
        requests = []
        def failed(request):
            requests.append(request)
            return httpx.Response(429, text='private-fixture')
        teacher = Teacher(self.settings, httpx.MockTransport(failed))
        with self.assertRaisesRegex(RuntimeError, 'HTTP 429'):
            list(teacher.stream(self.messages('image'), 10, modality='image'))
        with self.assertRaisesRegex(RuntimeError, '额度'):
            list(teacher.stream(self.messages('image'), 10, modality='image'))
        self.assertEqual(len(requests), 1); self.assertEqual(teacher.calls, 1)


if __name__ == '__main__': unittest.main()
