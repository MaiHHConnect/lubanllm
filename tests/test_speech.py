import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from luban.speech import MAX_AUDIO_BYTES, SpeechRuntime, decode_wav


def wav(samples=None, bits=16, encoding=1, rate=16000, channels=1):
    if samples is None:
        samples = b"\0\0" * 1600
    width = bits // 8
    fmt = struct.pack("<HHIIHH", encoding, channels, rate, rate * channels * width, channels * width, bits)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
    body += b"\0" * (len(samples) % 2)
    return b"RIFF" + struct.pack("<I", len(body)) + body


class SpeechTests(unittest.TestCase):
    def test_pcm_widths_have_same_normalized_signal(self):
        for bits, pack, low, high in [(8, "B", 0, 255), (16, "h", -32768, 32767), (32, "i", -2147483648, 2147483647)]:
            data = struct.pack("<" + pack * 1600, *([low, high] * 800))
            audio, duration = decode_wav(wav(data, bits))
            self.assertEqual(len(audio), 1600)
            self.assertEqual(audio[0], -1)
            self.assertGreater(audio[1], 0.99)
            self.assertEqual(duration, 0.1)
        audio, _ = decode_wav(wav((b"\0\0\x80\xff\xff\x7f") * 800, 24))
        self.assertEqual(audio[0], -1)
        self.assertGreater(audio[1], 0.99)

    def test_float32_and_resample(self):
        audio, duration = decode_wav(wav(struct.pack("<f", 0.25) * 4800, 32, 3, 48000))
        self.assertEqual(len(audio), 1600)
        self.assertEqual(duration, 0.1)
        self.assertTrue((audio == 0.25).all())

    def test_malformed_and_unbounded_input_rejected_before_loading(self):
        runtime = SpeechRuntime("/nonexistent")
        load = runtime._load = Mock()
        for data in [b"", b"x" * (MAX_AUDIO_BYTES + 1), wav()[:-1], wav(channels=2),
                     wav(rate=0), wav(b"\0" * (16000 * 2 * 31)), wav(b"\0"),
                     wav(struct.pack("<f", float("nan")) * 1600, 32, 3),
                     wav(struct.pack("<f", 1.1) * 1600, 32, 3)]:
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                runtime.transcribe_bytes(data)
        load.assert_not_called()

    def test_success_and_error_release_all_model_references(self):
        runtime = SpeechRuntime("/nonexistent")
        processor = Mock()
        processor.apply_transcription_request.return_value = {"input_ids": torch.tensor([[1, 2]])}
        processor.decode.return_value = {"language": "Chinese", "transcription": "你好。"}
        model = Mock()
        model.generation_config = SimpleNamespace(eos_token_id=9)
        model.generate.return_value = torch.tensor([[1, 2, 4, 9]])
        def load():
            runtime.model, runtime.processor = model, processor
        runtime._load = load
        result = runtime.transcribe_bytes(wav(), "zh")
        self.assertEqual(result["text"], "你好。")
        self.assertEqual(result["language"], "Chinese")
        self.assertIsNone(runtime.model)
        self.assertIsNone(runtime.processor)
        self.assertEqual(runtime.status()["state"], "unloaded")
        model.generate.side_effect = RuntimeError("original failure")
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            runtime.transcribe_bytes(wav())
        self.assertIsNone(runtime.model)
        self.assertIsNone(runtime.processor)
        self.assertEqual(runtime.error, "original failure")

    def test_truncated_output_is_an_error(self):
        runtime = SpeechRuntime("/nonexistent")
        processor = Mock()
        processor.apply_transcription_request.return_value = {"input_ids": torch.tensor([[1]])}
        model = Mock()
        model.generation_config = SimpleNamespace(eos_token_id=9)
        model.generate.return_value = torch.zeros((1, 513), dtype=torch.int64)
        def load():
            runtime.model, runtime.processor = model, processor
        runtime._load = load
        with self.assertRaisesRegex(RuntimeError, "输出上限"):
            runtime.transcribe_bytes(wav())
        processor.decode.assert_not_called()
        self.assertIsNone(runtime.model)

    def test_invalid_language_does_not_load_weights(self):
        runtime = SpeechRuntime("/nonexistent")
        runtime._load = Mock()
        for language in ["unsupported", [], 17]:
            with self.assertRaises(ValueError):
                runtime.transcribe_bytes(wav(), language)
        runtime._load.assert_not_called()

    def test_failed_partial_load_releases_processor(self):
        runtime = SpeechRuntime("/nonexistent")
        def load():
            runtime.processor = Mock()
            raise RuntimeError("bad weights")
        runtime._load = load
        with self.assertRaisesRegex(RuntimeError, "bad weights"):
            runtime.transcribe_bytes(wav())
        self.assertIsNone(runtime.processor)

    def test_missing_weights_never_trigger_network(self):
        runtime = SpeechRuntime("/nonexistent")
        with self.assertRaisesRegex(RuntimeError, "找不到"):
            runtime.transcribe_bytes(wav())
        self.assertFalse(runtime.status()["available"])


if __name__ == "__main__":
    unittest.main()
