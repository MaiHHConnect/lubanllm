"""Compaction boundaries with an offline tokenizer/model fixture, no model claims."""
import copy
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock

from luban.chat_context import compact_messages, context_usage


class ChatContextTests(unittest.TestCase):
    def runtime(self, limit=1000):
        runtime = SimpleNamespace(active_core=None, branch=SimpleNamespace(expert=object()),
            memory=SimpleNamespace(record_usage=Mock(), event=Mock()), _ready=Mock())
        runtime.context_status = lambda: {"current_limit": 262144, "memory_estimate_limit": limit}
        runtime.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw:
            list(range(sum(len(message["content"]) // 4 + 4 for message in messages))))
        def generate(messages, emit, stop, **kw):
            self.assertFalse(kw["allow_reference"])
            self.assertEqual(kw["thinking"], "off")
            emit({"content": "目标：完成任务。约束：保留原记录。"})
            return {"prompt_tokens": 700, "completion_tokens": 12, "total_tokens": 712,
                    "finish_reason": "stop", "elapsed_seconds": 1}
        runtime.generate = Mock(side_effect=generate)
        return runtime

    def history(self):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": "旧资料" * 520 if i < 2 else "最近消息"} for i in range(7)]

    def test_threshold_uses_actual_memory_limit_and_keeps_recent_originals(self):
        runtime, messages = self.runtime(), self.history()
        original, expert = copy.deepcopy(messages), runtime.branch.expert
        compressed, status, event = compact_messages(runtime, messages, threading.Event())
        self.assertEqual(messages, original)
        self.assertIs(runtime.branch.expert, expert)
        self.assertIsNotNone(event)
        self.assertEqual(event["prefix_count"], 2)
        self.assertEqual(compressed[-5:], messages[-5:])
        self.assertLess(status["input_tokens"], 750)
        self.assertEqual(status["limit"], 1000)
        runtime.memory.record_usage.assert_called_once()
        self.assertNotIn("summary", runtime.memory.event.call_args.args[1])

    def test_below_threshold_never_generates_or_bills(self):
        runtime = self.runtime(10000)
        messages = self.history()
        result, _, event = compact_messages(runtime, messages, threading.Event())
        self.assertIs(result, messages)
        self.assertIsNone(event)
        runtime.generate.assert_not_called()
        runtime.memory.record_usage.assert_not_called()

    def test_force_compacts_without_erasing_a_short_recent_conversation(self):
        runtime, messages = self.runtime(10000), self.history()
        self.assertIsNotNone(compact_messages(runtime, messages, threading.Event(), force=True)[2])
        runtime.generate.reset_mock()
        short = messages[-3:]
        self.assertIsNone(compact_messages(runtime, short, threading.Event(), force=True)[2])
        runtime.generate.assert_not_called()

    def test_failed_or_truncated_summary_keeps_original_and_restores_expert(self):
        for reason in ("length", "error"):
            runtime, messages = self.runtime(), self.history()
            original, expert = copy.deepcopy(messages), runtime.branch.expert
            if reason == "error":
                runtime.generate.side_effect = ValueError("fixture failed")
            else:
                def truncated(messages, emit, stop, **kw):
                    emit({"content": "不完整"})
                    return {"finish_reason": "length"}
                runtime.generate.side_effect = truncated
            with self.assertRaises(ValueError):
                compact_messages(runtime, messages, threading.Event())
            self.assertEqual(messages, original)
            self.assertIs(runtime.branch.expert, expert)

    def test_system_instructions_remain_separate_from_lossy_summary(self):
        runtime = self.runtime()
        messages = [{"role": "system", "content": "系统约束"}] + self.history()
        result, _, event = compact_messages(runtime, messages, threading.Event())
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[1]["role"], "user")
        self.assertNotIn(messages[0], runtime.generate.call_args.args[0])

    def test_actual_ui_sse_saves_compaction_before_answer_and_manual_does_not_answer(self):
        from test_app import AppTests, events
        app = AppTests(methodName="test_parameter_validation_prevents_any_generation")
        app.setUp()
        self.addCleanup(app.doCleanups)
        app.login()
        probe = self.runtime()
        app.runtime._ready = probe._ready
        app.runtime.tokenizer = probe.tokenizer
        app.runtime.context_status = probe.context_status
        app.runtime.branch = probe.branch
        original = app.runtime.generate
        def generate(messages, emit, stop, **kwargs):
            if kwargs.get("allow_reference") is False:
                return probe.generate(messages, emit, stop, **kwargs)
            return original(messages, emit, stop, **kwargs)
        app.runtime.generate = generate
        for manual in (False, True):
            result = app.client.post("/api/chat", json={"messages": self.history(), "auto_compact": True, "compact_only": manual})
            self.assertEqual(result.status_code, 200)
            stream = events(result)
            self.assertFalse(any(event["type"] == "error" for event in stream), stream)
            self.assertEqual(sum(event["type"] == "compaction" for event in stream), 1)
            self.assertEqual(stream[-1]["type"], "done")
            if manual:
                self.assertTrue(stream[-1]["compact_only"])
                self.assertFalse(any(event["type"] == "delta" for event in stream))
        self.assertEqual(len(app.runtime.calls), 1)
        self.assertEqual(app.runtime.memory.samples(), [])

    def test_zero_memory_rejects_before_generation(self):
        runtime = self.runtime(0)
        with self.assertRaisesRegex(ValueError, "足够"):
            context_usage(runtime, self.history())
        runtime.generate.assert_not_called()

    def test_stopped_runtime_reports_that_start_is_required(self):
        runtime = self.runtime()
        runtime.state = "stopped"
        with self.assertRaisesRegex(RuntimeError, "启动模型"):
            context_usage(runtime, self.history())
        runtime.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
