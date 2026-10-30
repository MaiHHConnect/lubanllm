"""CPU-only IPC and admission regressions for retained MLX prompt state."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from luban.native_engine import NativeEngine, NativeEngineError
from luban.runtime import Runtime


def engine_fixture(*, supported=True):
    engine = NativeEngine('/unused/worker', '/unused/metadata', 'a' * 64,
        precision='mlx-q4', mlx_model={'path': '/unused/model', 'sha256': 'b' * 64, 'weight_bytes': 1024})
    engine.loaded = {'mlx_cache_reuse': supported}
    engine.start = Mock(return_value=engine.loaded)
    engine._submit = Mock()
    engine._next_event = Mock()
    return engine


def acknowledge(engine):
    engine._next_event.side_effect = lambda _: {'id': str(engine._request_id), 'event': 'cache_cleared'}


class NativePrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine_fixture()
        self.addCleanup(self.engine.close)

    def test_exact_prefix_is_kept_without_ipc(self):
        self.engine._prefix_tokens = [11, 12]
        self.engine.prepare_prompt_cache([11, 12, 13], cache_reuse=True)
        self.assertEqual(self.engine._prefix_tokens, [11, 12])
        self.engine._submit.assert_not_called()
        self.engine._next_event.assert_not_called()

    def test_resident_ledger_requires_exact_prefix_and_is_discarded_on_clear(self):
        self.engine._prefix_tokens = [11, 12]
        self.engine._prefix_bytes = 512
        self.engine._prefix_step = 64
        self.assertEqual(self.engine.prompt_cache([11, 12, 13], cache_reuse=True),
                         {'tokens': 2, 'bytes': 512, 'prefill_step': 64})
        for tokens, reuse in [([11, 12, 13], False), ([11, 12], True), ([11, 15, 13], True)]:
            self.assertIsNone(self.engine.prompt_cache(tokens, cache_reuse=reuse))
        acknowledge(self.engine)
        self.engine.prepare_prompt_cache([], cache_reuse=False)
        self.assertEqual(self.engine._prefix_bytes, 0)
        self.assertIsNone(self.engine._prefix_step)

    def test_miss_shortening_and_nonextend_clear_with_ack(self):
        for tokens in ([10, 12, 13], [11, 15, 13], [11], [11, 12]):
            with self.subTest(tokens=tokens):
                self.engine._prefix_tokens = [11, 12]
                acknowledge(self.engine)
                self.engine._submit.reset_mock()
                self.engine.prepare_prompt_cache(tokens, cache_reuse=True)
                request, deadline, stop = self.engine._submit.call_args.args
                self.assertEqual(request, {'id': str(self.engine._request_id), 'op': 'clear_cache'})
                self.assertGreater(deadline, 0)
                self.assertIsInstance(stop, threading.Event)
                self.assertEqual(self.engine._prefix_tokens, [])

    def test_no_reuse_clears_even_a_matching_prefix(self):
        self.engine._prefix_tokens = [11, 12]
        acknowledge(self.engine)
        self.engine.prepare_prompt_cache([11, 12, 13], cache_reuse=False)
        self.assertEqual(self.engine._prefix_tokens, [])
        self.assertEqual(self.engine._submit.call_args.args[0]['op'], 'clear_cache')

    def test_empty_slot_does_not_start_or_submit(self):
        for reuse in (False, True):
            self.engine.prepare_prompt_cache([11, 12], cache_reuse=reuse)
        self.engine.start.assert_not_called()
        self.engine._submit.assert_not_called()
        self.engine._next_event.assert_not_called()

    def test_bad_ack_closes_worker_and_discards_ledger(self):
        for event in ({'id': 'wrong', 'event': 'cache_cleared'},
                      {'id': '1', 'event': 'result'}):
            with self.subTest(event=event):
                engine = engine_fixture()
                engine._prefix_tokens = [11, 12]
                engine._next_event.return_value = event
                with patch.object(engine, 'close', wraps=engine.close) as close:
                    with self.assertRaisesRegex(NativeEngineError, 'acknowledgement mismatch'):
                        engine.prepare_prompt_cache([5, 6], cache_reuse=True)
                    close.assert_called_once()
                self.assertIsNone(engine.process)
                self.assertEqual(engine._prefix_tokens, [])

    def test_transport_failure_closes_and_discards_ledger(self):
        self.engine._prefix_tokens = [11, 12]
        self.engine._submit.side_effect = OSError('broken pipe')
        with patch.object(self.engine, 'close', wraps=self.engine.close) as close:
            with self.assertRaisesRegex(NativeEngineError, 'broken pipe'):
                self.engine.prepare_prompt_cache([5, 6], cache_reuse=True)
            close.assert_called_once()
        self.assertEqual(self.engine._prefix_tokens, [])

    def generate_result(self, engine, *, capacity=2, reuse=True, size=0):
        engine._next_event.side_effect = [
            {'id': str(engine._request_id + 1), 'event': 'token', 'token': 7},
            {'id': str(engine._request_id + 1), 'event': 'result', 'tokens': [7],
             'generated_tokens': 1, 'finish_reason': 'length', 'cache_capacity': capacity,
             'cached_prompt_tokens': 0, 'prefix_cache_bytes': size},
        ]
        return engine.generate([11, 12, 13], 1, [9], 0, lambda _: None,
                               threading.Event(), cache_reuse=reuse)

    def test_generation_records_only_acknowledged_input_boundary(self):
        answer = self.generate_result(self.engine, size=512)
        self.assertEqual(answer['tokens'], [7])
        self.assertEqual(self.engine._prefix_tokens, [11, 12])
        self.assertEqual(self.engine.prompt_cache([11, 12, 13], cache_reuse=True),
                         {'tokens': 2, 'bytes': 512, 'prefill_step': 512})
        # A generated token is never inferred to belong to the saved prefix.
        self.assertNotIn(7, self.engine._prefix_tokens)
        self.engine._submit.reset_mock()
        self.engine.prepare_prompt_cache([11, 12, 13], cache_reuse=True)
        self.engine._submit.assert_not_called()

    def test_unsupported_worker_never_creates_or_clears_a_slot(self):
        engine = engine_fixture(supported=False)
        self.addCleanup(engine.close)
        self.generate_result(engine)
        self.assertEqual(engine._prefix_tokens, [])
        engine._submit.reset_mock()
        engine.prepare_prompt_cache([11, 12, 13], cache_reuse=True)
        engine._submit.assert_not_called()

    def test_invalid_retained_boundary_closes_worker(self):
        for capacity, reuse in ((True, True), (-1, True), (3, True), (4, True), (1, False)):
            with self.subTest(capacity=capacity, reuse=reuse):
                engine = engine_fixture()
                with patch.object(engine, 'close', wraps=engine.close) as close:
                    with self.assertRaisesRegex(NativeEngineError, 'retained prefix count'):
                        self.generate_result(engine, capacity=capacity, reuse=reuse)
                    close.assert_called_once()
                self.assertEqual(engine._prefix_tokens, [])

    def test_invalid_retained_bytes_close_worker(self):
        for size, capacity in [(True, 2), (-1, 2), (1.5, 2), (512, 0)]:
            with self.subTest(size=size, capacity=capacity):
                engine = engine_fixture()
                with self.assertRaisesRegex(NativeEngineError, 'retained prefix bytes'):
                    self.generate_result(engine, size=size, capacity=capacity)
                self.assertEqual(engine._prefix_bytes, 0)
                self.assertIsNone(engine._prefix_step)


class RuntimePrefixAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine_fixture()
        self.addCleanup(self.engine.close)
        self.engine._prefix_tokens = [11, 12]
        acknowledge(self.engine)
        self.runtime = Runtime.__new__(Runtime)
        self.runtime.native_engine = self.engine
        self.runtime.device = 'cpu'
        self.runtime.tokenizer = SimpleNamespace(apply_chat_template=Mock(
            return_value=torch.tensor([[21, 22, 23]], dtype=torch.long)))
        self.status = {'current_limit': 128, 'linear_bytes_per_token': 1,
                       'quadratic_bytes_per_token_squared': 0, 'memory_budget_bytes': 1024}
        self.runtime.context_status = Mock(return_value=self.status)

    def test_miss_is_evicted_before_status_and_real_memory_admission(self):
        def status(*, shallow):
            self.assertFalse(shallow)
            self.assertEqual(self.engine._prefix_tokens, [])
            return self.status

        self.runtime.context_status.side_effect = status
        from luban.context import check_input
        with patch('luban.runtime.check_input', wraps=check_input) as admission:
            ids = self.runtime._tokens([], output_reserve=16, cache_reuse=True)
        self.assertEqual(ids.tolist(), [[21, 22, 23]])
        admission.assert_called_once_with(3, 16, self.status)
        self.assertEqual(self.engine._prefix_tokens, [])

    def test_matching_prefix_is_retained_through_admission(self):
        self.runtime.tokenizer.apply_chat_template.return_value = torch.tensor([[11, 12, 13]])
        ids = self.runtime._tokens([], cache_reuse=True)
        self.assertEqual(ids.tolist(), [[11, 12, 13]])
        self.assertEqual(self.engine._prefix_tokens, [11, 12])
        self.engine._submit.assert_not_called()

    def test_no_reuse_evicts_before_admission_even_when_matching(self):
        self.runtime.tokenizer.apply_chat_template.return_value = torch.tensor([[11, 12, 13]])
        self.runtime._tokens([], cache_reuse=False)
        self.assertEqual(self.engine._prefix_tokens, [])
        self.assertEqual(self.engine._submit.call_args.args[0]['op'], 'clear_cache')

    def test_matching_cache_is_released_and_memory_rechecked_when_needed(self):
        self.runtime.tokenizer.apply_chat_template.return_value = torch.tensor([[11, 12, 13]])
        low = {**self.status, 'current_limit': 2}
        self.runtime.context_status.side_effect = [low, self.status]
        self.assertEqual(self.runtime._tokens([], cache_reuse=True).tolist(), [[11, 12, 13]])
        self.assertEqual(self.engine._prefix_tokens, [])
        self.assertEqual(self.runtime.context_status.call_count, 2)
        self.assertEqual(self.engine._submit.call_count, 1)

    def test_eviction_does_not_bypass_a_still_insufficient_memory_limit(self):
        self.runtime.tokenizer.apply_chat_template.return_value = torch.tensor([[11, 12, 13]])
        self.status['current_limit'] = 2
        with self.assertRaisesRegex(ValueError, '超过当前上下文'):
            self.runtime._tokens([], cache_reuse=True)
        self.assertEqual(self.engine._prefix_tokens, [])
        self.assertEqual(self.runtime.context_status.call_count, 2)

    def test_shallow_keeps_cache_and_still_checks_its_own_context(self):
        self.status['current_limit'] = 2
        with self.assertRaisesRegex(ValueError, '超过当前上下文'):
            self.runtime._tokens([], shallow=True)
        self.assertEqual(self.engine._prefix_tokens, [11, 12])
        self.engine._submit.assert_not_called()
        self.runtime.context_status.assert_called_once_with(shallow=True)

    def test_bad_eviction_ack_prevents_admission(self):
        self.engine._next_event.side_effect = None
        self.engine._next_event.return_value = {'id': 'wrong', 'event': 'cache_cleared'}
        with patch('luban.runtime.check_input') as admission:
            with self.assertRaises(NativeEngineError):
                self.runtime._tokens([], cache_reuse=True)
        admission.assert_not_called()
        self.runtime.context_status.assert_not_called()
        self.assertEqual(self.engine._prefix_tokens, [])

    def test_empty_slot_has_no_extra_eviction_command(self):
        self.engine._prefix_tokens = []
        self.runtime._tokens([], cache_reuse=True)
        self.engine._submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
