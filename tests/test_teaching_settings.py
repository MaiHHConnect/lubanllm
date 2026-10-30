"""Offline teacher scope validation; no requests, models or training."""
import json
from pathlib import Path
import tempfile
import unittest
from luban.settings import Settings


class TeachingSettingsTests(unittest.TestCase):
    def test_legacy_default_empty_canonical_order_and_reopen(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home, "settings.json")
            path.write_text(json.dumps({"teacher": {"model": "offline"}}))
            settings = Settings(home)
            self.assertEqual(settings.snapshot()["teacher"]["teaching_modalities"], ["text"])
            self.assertEqual(settings.snapshot()["teacher"]["session_call_limit"], 0)
            for selected, expected in (([], []), (["audio", "text", "video", "image"], ["text", "image", "video", "audio"]),
                                       (["audio", "image"], ["image", "audio"])):
                settings.update({"teacher": {"teaching_modalities": selected}})
                self.assertEqual(Settings(home).public()["teacher"]["teaching_modalities"], expected)
                self.assertEqual(settings.snapshot()["teacher"]["model"], "offline")
            settings.update({"teacher": {"name": "Teacher"}})
            self.assertEqual(settings.snapshot()["teacher"]["teaching_modalities"], ["image", "audio"])

    def test_invalid_update_never_changes_settings_or_credentials(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home)
            saved = settings.path.read_bytes(), settings.secret_path.read_bytes()
            for value in (None, "text", True, 1, {}, ["unknown"], ["TEXT"], ["text", "text"], [1], [None], [[]], [{}]):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    settings.update({"teacher": {"teaching_modalities": value, "api_key": "must-not-be-saved"}})
                self.assertEqual((settings.path.read_bytes(), settings.secret_path.read_bytes()), saved)
                self.assertEqual(settings.snapshot()["teacher"]["teaching_modalities"], ["text"])

    def test_invalid_saved_scope_is_rejected_without_replacing_original(self):
        for value in (None, "image", ["text", "text"], ["operation"], [True]):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as home:
                path = Path(home, "settings.json")
                raw = json.dumps({"teacher": {"teaching_modalities": value}})
                path.write_text(raw)
                with self.assertRaises(ValueError): Settings(home)
                self.assertEqual(path.read_text(), raw)


if __name__ == "__main__": unittest.main()
