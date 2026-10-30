"""Tools enter the real template and share the actual input admission budget."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from luban.runtime import Runtime


class RuntimeToolTests(unittest.TestCase):
    def test_tool_template_tokens_share_context_admission(self):
        runtime = Runtime.__new__(Runtime)
        runtime.device = 'cpu'
        runtime.tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value=torch.ones((1, 24), dtype=torch.long)))
        runtime.context_status = Mock(return_value={'current_limit': 25, 'linear_bytes_per_token': 1, 'quadratic_bytes_per_token_squared': 0, 'memory_budget_bytes': 1024})
        tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
        messages = [{'role': 'user', 'content': 'test'}]
        self.assertEqual(runtime._tokens(messages, tools=tools).shape[1], 24)
        self.assertEqual(runtime.tokenizer.apply_chat_template.call_args.kwargs['tools'], tools)
        runtime.tokenizer.apply_chat_template.return_value = torch.ones((1, 26), dtype=torch.long)
        with self.assertRaises(ValueError):
            runtime._tokens(messages, tools=tools)
