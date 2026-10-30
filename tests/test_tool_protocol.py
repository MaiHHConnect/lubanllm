"""CPU protocol checks; generated calls are parsed, never executed."""
import copy
import json
import unittest

from luban.tool_protocol import (ToolOutputBuffer, normalize_tool_messages, parse_tool_output,
                                 prepare_tool_prompt, validate_tool_options, validate_tools)


def fixture_tools():
    return validate_tools({"tools": [{"type": "function", "function": {"name": "weather",
        "description": "Read the weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 7},
            "options": {"type": "object", "properties": {"units": {"enum": ["c", "f"]}}}},
            "required": ["city"], "additionalProperties": False}}}]})


def xml(city="杭州", extra=""):
    return ("<tool_call>\n<function=weather>\n<parameter=city>\n" + city
        + "\n</parameter>\n" + extra + "</function>\n</tool_call>")


def call(call_id="call_a", arguments='{"city":"杭州"}', name="weather"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class ToolProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tools = fixture_tools()

    def test_tools_copied_and_default_schema(self):
        body = {"tools": [{"type": "function", "function": {"name": "read"}}]}
        before = copy.deepcopy(body)
        self.assertEqual(validate_tools(body)[0]["function"]["parameters"]["type"], "object")
        self.assertEqual(before, body)

    def test_invalid_tools_rejected(self):
        for tools in (None, {}, [1], [{"type": "web_search"}], self.tools * 2):
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                validate_tools({"tools": tools})

    def test_tool_options(self):
        self.assertEqual(validate_tool_options({}, []), ("none", True))
        self.assertEqual(validate_tool_options({}, self.tools), ("auto", True))
        self.assertEqual(validate_tool_options({"tool_choice": {"type": "function",
            "function": {"name": "weather"}}, "parallel_tool_calls": False}, self.tools), ("weather", False))
        for body in ({"tool_choice": "bad"}, {"parallel_tool_calls": 1},
                     {"tool_choice": {"type": "function", "function": {"name": "unknown"}}}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                validate_tool_options(body, self.tools)
        with self.assertRaises(ValueError):
            validate_tool_options({"tool_choice": "required"}, [])

    def test_tool_prompt_constraints_count_as_actual_system_text(self):
        history = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "weather?"}]
        before = copy.deepcopy(history)
        self.assertEqual(prepare_tool_prompt(history, "auto", True), history)
        required = prepare_tool_prompt(history, "required", False)
        self.assertIn("must call", required[0]["content"])
        self.assertIn("at most one", required[0]["content"])
        self.assertEqual(history, before)
        self.assertIn("weather", prepare_tool_prompt(history, "weather", True)[0]["content"])
        self.assertIn("Do not call", prepare_tool_prompt(history[1:], "none", True)[0]["content"])

    def test_history_roundtrip_and_no_mutation(self):
        history = [{"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [call()]},
            {"role": "tool", "tool_call_id": "call_a", "content": ""}]
        original = copy.deepcopy(history)
        normalized = normalize_tool_messages(history)
        self.assertEqual(history, original)
        self.assertEqual(normalized[1]["content"], "")
        self.assertEqual(normalized[1]["tool_calls"][0]["function"]["arguments"], {"city": "杭州"})
        self.assertEqual(normalized[-1]["content"], "")

    def test_history_multi_tools_allow_results_in_reverse_order(self):
        history = [{"role": "assistant", "content": None, "tool_calls": [call("a"), call("b")]},
            {"role": "tool", "tool_call_id": "b", "content": "B"},
            {"role": "tool", "tool_call_id": "a", "name": "weather", "content": "A"}]
        self.assertEqual(len(normalize_tool_messages(history)), 3)

    def test_history_rejects_dangling_duplicates_wrong_names_and_nontext(self):
        assistant = {"role": "assistant", "content": None, "tool_calls": [call()]}
        result = {"role": "tool", "tool_call_id": "call_a", "content": "ok"}
        cases = [[assistant], [result], [assistant, result, result],
            [assistant, {"role": "user", "content": "skip"}, result],
            [assistant, {**result, "name": "wrong"}], [assistant, {**result, "content": []}],
            [{**assistant, "tool_calls": [call(), call()]}],
            [{"role": "user", "content": "x", "tool_calls": [call()]}],
            [{"role": "user", "content": "x", "tool_call_id": "x"}],
            [{**assistant, "tool_calls": [call(arguments='[]')]}],
            [{**assistant, "tool_calls": [call(arguments='{"x":NaN}')]}]]
        for history in cases:
            with self.subTest(history=history), self.assertRaises(ValueError):
                normalize_tool_messages(history)

    def test_actual_qwen_xml_format_and_typed_parameters(self):
        text = "I will look it up.\n\n" + xml(extra=("<parameter=days>\n3\n</parameter>\n"
            '<parameter=options>\n{"units":"c"}\n</parameter>\n'))
        content, calls = parse_tool_output(text, self.tools)
        self.assertEqual(content, "I will look it up.")
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "杭州", "days": 3, "options": {"units": "c"}})

    def test_string_parameters_remain_strings(self):
        for text in ("123", "true", "null", '["a"]'):
            _, calls = parse_tool_output(xml(text), self.tools)
            self.assertEqual(json.loads(calls[0]["function"]["arguments"])["city"], text)

    def test_json_variants(self):
        raw = {"name": "weather", "arguments": {"city": "Shanghai"}}
        for text in (json.dumps(raw), "<tool_call>" + json.dumps(raw) + "</tool_call>",
            json.dumps({"tool_calls": [{"type": "function", "function": raw}]}),
            json.dumps({"type": "function", "function": {**raw, "arguments": '{"city":"Shanghai"}'}})):
            with self.subTest(text=text):
                content, calls = parse_tool_output(text, self.tools)
                self.assertEqual(content, "")
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Shanghai"})

    def test_unknown_invalid_incomplete_calls_do_not_dispatch(self):
        for text in (xml().replace("weather", "other"), xml().replace("</tool_call>", ""),
            xml() + "trailing words", xml() + "notes" + xml(),
            '<tool_call>{"name":"weather","arguments":bad}</tool_call>',
            '<tool_call>{"name":"weather","arguments":{"city":"a","city":"b"}}</tool_call>',
            xml(extra="<parameter=days>\n9\n</parameter>\n"),
            xml(extra="<parameter=unknown>\n1\n</parameter>\n"),
            xml(extra="<parameter=city>\nElsewhere\n</parameter>\n"),
            '<tool_call><function=weather></function></tool_call>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_tool_output(text, self.tools)

    def test_choice_and_parallel_enforced_without_fabricating_calls(self):
        with self.assertRaises(ValueError):
            parse_tool_output("I cannot use that tool", self.tools, "required")
        with self.assertRaises(ValueError):
            parse_tool_output(xml(), self.tools, "none")
        with self.assertRaises(ValueError):
            parse_tool_output(xml(), self.tools, "other")
        with self.assertRaises(ValueError):
            parse_tool_output(xml() + "\n" + xml(), self.tools, parallel=False)
        self.assertEqual(len(parse_tool_output(xml() + "\n" + xml(), self.tools)[1]), 2)
        self.assertEqual(parse_tool_output("ordinary answer", self.tools), ("ordinary answer", []))

    def test_every_possible_two_chunk_boundary(self):
        text = "Let me check.\n" + xml(extra="<parameter=days>\n2\n</parameter>\n")
        for split in range(len(text) + 1):
            buffer = ToolOutputBuffer(self.tools)
            self.assertIsNone(buffer.feed(text[:split]))
            self.assertIsNone(buffer.feed(text[split:]))
            content, calls = buffer.finish()
            self.assertEqual(content, "Let me check.")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"])["days"], 2)

    def test_buffer_bound_and_incomplete_finish(self):
        buffer = ToolOutputBuffer(self.tools, max_chars=5)
        with self.assertRaises(ValueError):
            buffer.feed("123456")
        buffer = ToolOutputBuffer(self.tools)
        buffer.feed("<tool_call>\n<function=weather>")
        with self.assertRaises(ValueError):
            buffer.finish()

    def test_raw_stream_keeps_literal_thought_tags_inside_xml_arguments(self):
        raw = '<think>Choose the weather tool.</think>Let me check.\n' + xml("<think>杭州</think>")
        for split in range(len(raw) + 1):
            buffer = ToolOutputBuffer(self.tools)
            buffer.feed(raw[:split])
            buffer.feed(raw[split:])
            visible, calls = buffer.finish()
            self.assertEqual(visible, "Let me check.")
            self.assertEqual(buffer.reasoning, "Choose the weather tool.")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"])["city"], "<think>杭州</think>")

    def test_raw_stream_keeps_literal_thought_tags_inside_json_arguments(self):
        call_json = json.dumps({"name": "weather", "arguments": {"city": "<think>上海</think>"}}, ensure_ascii=False)
        for raw in (call_json, "<think>Pick a tool.</think>\n" + call_json,
                    '<think>Pick a tool.</think>\n<tool_call>' + call_json + '</tool_call>'):
            buffer = ToolOutputBuffer(self.tools)
            for char in raw:
                buffer.feed(char)
            visible, calls = buffer.finish()
            self.assertEqual(visible, "")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"])["city"], "<think>上海</think>")
            self.assertNotIn("上海", buffer.reasoning)

    def test_tool_syntax_inside_prefix_thought_is_not_dispatched(self):
        buffer = ToolOutputBuffer(self.tools)
        buffer.feed('<think>Example: ' + xml() + '</think>Final answer.')
        self.assertEqual(buffer.finish(), ("Final answer.", []))
        self.assertIn("<tool_call>", buffer.reasoning)

    def test_recursive_schema_refs_and_union(self):
        tools = validate_tools({"tools": [{"type": "function", "function": {"name": "weather",
            "parameters": {"type": "object", "$defs": {"cities": {"type": "array", "items": {"type": "string"},
                "minItems": 1}}, "properties": {"cities": {"$ref": "#/$defs/cities"},
                "days": {"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 1}]}},
                "required": ["cities"], "additionalProperties": False}}}]})
        self.assertEqual(len(parse_tool_output('<tool_call>{"name":"weather","arguments":'
            '{"cities":["A"],"days":null}}</tool_call>', tools)[1]), 1)
        with self.assertRaises(ValueError):
            parse_tool_output('<tool_call>{"name":"weather","arguments":{"cities":[1]}}</tool_call>', tools)

    def test_unsupported_schema_assertions_and_remote_refs_rejected(self):
        for schema in ({"type": "object", "dependentRequired": {}},
            {"type": "object", "properties": {"x": {"$ref": "https://example.com/schema"}}},
            {"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}},
            {"type": "object", "properties": {"x": {"pattern": "["}}},
            {"type": "object", "properties": {"x": {"minLength": "1"}}}):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                validate_tools({"tools": [{"type": "function", "function": {"name": "weather", "parameters": schema}}]})

    def test_boolean_and_number_enum_are_distinct(self):
        tools = fixture_tools()
        tools[0]["function"]["parameters"]["properties"]["days"] = {"enum": [1]}
        with self.assertRaises(ValueError):
            parse_tool_output(xml(extra="<parameter=days>\ntrue\n</parameter>\n"), tools)


if __name__ == "__main__":
    unittest.main()
