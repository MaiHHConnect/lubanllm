"""Text-only function calling for the shipped Qwen template; never executes tools.

History uses dictionary arguments for Jinja. Public calls use JSON strings. Model
output is buffered until complete so invalid/truncated calls never reach clients.
"""
from __future__ import annotations

import copy
import json
import math
import re
import uuid

from .thinking import ThinkingStream


_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_ANNOTATIONS = {"title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
                "$schema", "$id", "$comment", "format"}
_SCHEMA_KEYS = _ANNOTATIONS | {"type", "properties", "required", "additionalProperties", "items", "enum", "const",
    "anyOf", "oneOf", "allOf", "not", "$ref", "$defs", "definitions", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties"}
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def _loads(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"工具 JSON 字段重复：{key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"工具 JSON 不允许 {value}")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ValueError("工具参数需要合法 JSON") from exc


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("工具函数名无效")
    return value


def _reference(schema, root):
    ref = schema.get("$ref")
    if ref is None:
        return schema
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ValueError("工具 schema 仅支持本地 $ref")
    target = root
    try:
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError) as exc:
        raise ValueError("工具 schema 的 $ref 不存在") from exc
    if not isinstance(target, (dict, bool)):
        raise ValueError("工具 schema 的 $ref 需要引用 schema")
    return target


def _check_schema(schema, root, depth=0):
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict) or depth > 64:
        raise ValueError("工具参数 schema 无效或嵌套过深")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        raise ValueError(f"工具 schema 暂不支持约束：{', '.join(sorted(unknown))}")
    if "$ref" in schema:
        _reference(schema, root)  # Resolution is bounded during value validation.
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not types or any(not isinstance(t, str) or t not in _TYPES for t in types):
            raise ValueError("工具 schema 的 type 无效")
    for key in ("properties", "$defs", "definitions"):
        if key in schema:
            if not isinstance(schema[key], dict):
                raise ValueError(f"工具 schema 的 {key} 需要对象")
            for child in schema[key].values():
                _check_schema(child, root, depth + 1)
    if "required" in schema and (not isinstance(schema["required"], list)
            or any(not isinstance(key, str) for key in schema["required"])):
        raise ValueError("工具 schema 的 required 需要字符串数组")
    for key in ("additionalProperties", "items", "not"):
        if key in schema:
            _check_schema(schema[key], root, depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise ValueError(f"工具 schema 的 {key} 需要非空数组")
            for child in schema[key]:
                _check_schema(child, root, depth + 1)
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("工具 schema 的 enum 需要非空数组")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise ValueError("工具 schema 的 uniqueItems 需要布尔值")
    for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"工具 schema 的 {key} 需要非负整数")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])
                or (key == "multipleOf" and schema[key] <= 0)):
            raise ValueError(f"工具 schema 的 {key} 需要有限数值")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (TypeError, re.error) as exc:
            raise ValueError("工具 schema 的 pattern 无效") from exc


def _same(a, b):
    # JSON boolean and number are different types (Python's True == 1 is not).
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)


def _validate(value, schema, root, path="arguments", depth=0):
    if depth > 64:
        raise ValueError("工具参数嵌套过深或 schema 循环引用")
    if schema is True:
        return
    if schema is False:
        raise ValueError(f"{path} 不允许此值")
    if "$ref" in schema:
        target = _reference(schema, root)
        _check_schema(target, root)
        _validate(value, target, root, path, depth + 1)
    finite_number = type(value) is int or (type(value) is float and math.isfinite(value))
    checks = {"null": value is None, "boolean": type(value) is bool, "string": isinstance(value, str),
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "integer": finite_number and int(value) == value, "number": finite_number}
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(checks[t] for t in types):
            raise ValueError(f"{path} 类型不符合工具 schema")
    if "enum" in schema and not any(_same(value, item) for item in schema["enum"]):
        raise ValueError(f"{path} 不在 enum 中")
    if "const" in schema and not _same(value, schema["const"]):
        raise ValueError(f"{path} 不符合 const")
    for key in ("anyOf", "oneOf", "allOf", "not"):
        if key not in schema:
            continue
        variants = [schema[key]] if key == "not" else schema[key]
        matches = 0
        for child in variants:
            try:
                _validate(value, child, root, path, depth + 1)
                matches += 1
            except ValueError:
                pass
        if not {"anyOf": matches > 0, "oneOf": matches == 1,
                "allOf": matches == len(variants), "not": matches == 0}[key]:
            raise ValueError(f"{path} 不符合 {key}")
    if isinstance(value, dict):
        if any(key not in value for key in schema.get("required", [])):
            raise ValueError(f"{path} 缺少必需工具参数")
        properties = schema.get("properties", {})
        for key, item in value.items():
            child = properties.get(key, schema.get("additionalProperties", True))
            _validate(item, child, root, f"{path}.{key}", depth + 1)
        lower, upper = "minProperties", "maxProperties"
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _validate(item, schema.get("items", True), root, f"{path}[{i}]", depth + 1)
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise ValueError(f"{path} 数组元素重复")
        lower, upper = "minItems", "maxItems"
    elif isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(f"{path} 不符合 pattern")
        lower, upper = "minLength", "maxLength"
    else:
        lower = upper = None
    if lower and ((lower in schema and len(value) < schema[lower])
                  or (upper in schema and len(value) > schema[upper])):
        raise ValueError(f"{path} 长度不符合工具 schema")
    if type(value) in (int, float):
        if not finite_number:
            raise ValueError(f"{path} 需要有限数值")
        if (("minimum" in schema and value < schema["minimum"])
                or ("maximum" in schema and value > schema["maximum"])
                or ("exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"])
                or ("exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"])
                or ("multipleOf" in schema and not math.isclose(value / schema["multipleOf"],
                    round(value / schema["multipleOf"]), rel_tol=1e-9, abs_tol=1e-9))):
            raise ValueError(f"{path} 数值不符合工具 schema")


def validate_tools(body):
    """Return a copied OpenAI function-tool list; unsupported assertions fail early.

    `format` is an annotation as in JSON Schema's default vocabulary; no network,
    remote references, credential access, or tool execution occurs here.
    """
    tools = body.get("tools", [])
    if not isinstance(tools, list) or len(tools) > 128:
        raise ValueError("tools 需要最多 128 个函数工具")
    result, names = [], set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("仅支持 function 类型工具")
        function = copy.deepcopy(tool["function"])
        name = _name(function.get("name"))
        if name in names:
            raise ValueError("工具函数名重复")
        names.add(name)
        if "description" in function and not isinstance(function["description"], str):
            raise ValueError("工具 description 需要文本")
        if "strict" in function and not isinstance(function["strict"], bool):
            raise ValueError("工具 strict 需要布尔值")
        schema = function.setdefault("parameters", {"type": "object", "properties": {}})
        _check_schema(schema, schema)
        if not isinstance(schema, dict) or schema.get("type", "object") != "object":
            raise ValueError("工具 parameters 需要 object schema")
        result.append({"type": "function", "function": function})
    return result


def validate_tool_options(body, tools):
    """Return (choice, parallel); named choice is the actual function name."""
    choice = body.get("tool_choice", "auto" if tools else "none")
    if isinstance(choice, dict):
        if choice.get("type") != "function" or not isinstance(choice.get("function"), dict):
            raise ValueError("tool_choice 需要指定 function")
        choice = _name(choice["function"].get("name"))
        if choice not in {tool["function"]["name"] for tool in tools}:
            raise ValueError("tool_choice 指定了未知工具")
    elif choice not in ("auto", "none", "required"):
        raise ValueError("tool_choice 应为 auto、none、required 或指定 function")
    if choice == "required" and not tools:
        raise ValueError("tool_choice required 需要 tools")
    parallel = body.get("parallel_tool_calls", True)
    if not isinstance(parallel, bool):
        raise ValueError("parallel_tool_calls 需要布尔值")
    return choice, parallel


def prepare_tool_prompt(messages, choice, parallel):
    """Copy history and add only requested tool constraints before tokenization."""
    result = copy.deepcopy(messages)
    instructions = []
    if choice == "required":
        instructions.append("You must call an available function to answer this request.")
    elif choice == "none":
        instructions.append("Do not call any function. Answer using text only.")
    elif choice != "auto":
        instructions.append(f"You must call the function {choice}. Do not call any other function.")
    if not parallel:
        instructions.append("Call at most one function in this response.")
    if instructions:
        constraint = "\n".join(instructions)
        if result and result[0].get("role") == "system":
            result[0]["content"] = result[0]["content"] + "\n\n" + constraint
        else:
            result.insert(0, {"role": "system", "content": constraint})
    return result


def _arguments(value):
    result = _loads(value) if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(result, dict):
        raise ValueError("工具 arguments 需要 JSON 对象")
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("工具 arguments 需要有限 JSON 对象") from exc
    return result


def normalize_tool_messages(messages):
    """Copy history and check tool call/result pairing; normal text stays untouched.

    Call before normal content validation. Empty tool results are valid; an
    assistant's null/empty content is valid only when it has function calls.
    Every tool call must receive one result before another non-tool message.
    """
    result, pending, seen = [], {}, set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("消息需要对象")
        message = copy.deepcopy(message)
        role = message.get("role")
        calls = message.get("tool_calls")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("工具结果引用未知、重复或已完成的 tool_call_id")
            if "name" in message and message["name"] != pending[call_id]:
                raise ValueError("工具结果 name 与调用不符")
            if calls is not None or not isinstance(message.get("content"), str):
                raise ValueError("工具结果仅支持文字 content")
            del pending[call_id]
        else:
            if pending:
                raise ValueError("assistant 工具调用缺少对应的 tool 结果")
            if "tool_call_id" in message:
                raise ValueError("仅 tool 消息可携带 tool_call_id")
            if calls is not None:
                if role != "assistant" or not isinstance(calls, list) or not calls or len(calls) > 128:
                    raise ValueError("tool_calls 需要 assistant 的非空调用列表")
                normalized = []
                for call in calls:
                    if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict):
                        raise ValueError("tool_calls 仅支持 function")
                    call_id = call.get("id")
                    if (not isinstance(call_id, str) or not call_id.strip() or len(call_id) > 256
                            or any(ord(c) < 33 for c in call_id) or call_id in seen):
                        raise ValueError("工具调用 id 无效或重复")
                    name = _name(call["function"].get("name"))
                    args = _arguments(call["function"].get("arguments"))
                    normalized.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args}})
                    seen.add(call_id)
                    pending[call_id] = name
                message["tool_calls"] = normalized
                if message.get("content") is None:
                    message["content"] = ""
        result.append(message)
    if pending:
        raise ValueError("assistant 工具调用缺少对应的 tool 结果")
    return result


def _xml_arguments(text, schema):
    args, end = {}, 0
    for match in re.finditer(r"<parameter=([^<>\s]+)>(.*?)</parameter>", text, re.S):
        if text[end:match.start()].strip():
            raise ValueError("工具 XML 参数格式无效")
        key, raw = match.groups()
        if key in args:
            raise ValueError("工具 XML 参数重复")
        # Template surrounds parameter values with one newline on each side.
        raw = raw.removeprefix("\n").removesuffix("\n")
        child = schema.get("properties", {}).get(key, {})
        if isinstance(child, dict) and "$ref" in child:
            child = _reference(child, schema)
        if isinstance(child, dict) and child.get("type") == "string":
            value = raw
        else:
            try:
                value = _loads(raw)
            except ValueError:
                value = raw
        args[key], end = value, match.end()
    if text[end:].strip():
        raise ValueError("工具 XML 参数不完整或无效")
    return args


def parse_tool_output(text, tools, choice="auto", parallel=True):
    """Return (visible text, public OpenAI calls), or ValueError for invalid calls.

    Accept shipped XML and JSON inside <tool_call>; bare JSON is recognized only
    when tools are available and the complete output is a tool-call object.
    """
    functions = {tool["function"]["name"]: tool["function"] for tool in tools}
    raw_calls, content = [], text
    if "<tool_call>" in text:
        first, end = text.index("<tool_call>"), 0
        content = text[:first].rstrip()
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text[first:], re.S):
            if text[first + end:first + match.start()].strip():
                raise ValueError("工具调用之间不允许额外文本")
            body = match.group(1).strip()
            if body.startswith("<function="):
                function = re.fullmatch(r"<function=([^<>\s]+)>\s*([\s\S]*?)\s*</function>", body)
                if not function:
                    raise ValueError("模型工具 XML 格式不完整")
                name, parameter_text = function.groups()
                if name not in functions:
                    raise ValueError(f"模型请求了未知工具：{name}")
                args = _xml_arguments(parameter_text, functions[name]["parameters"])
                raw_calls.append({"name": name, "arguments": args})
            else:
                raw_calls.append(_loads(body))
            end = match.end()
        if not raw_calls or text[first + end:].strip():
            raise ValueError("模型工具调用不完整或带有尾随文本")
    elif "<tool_call" in text or "</tool_call>" in text:
        raise ValueError("模型工具调用不完整")
    elif tools and text.strip().startswith("{"):
        try:
            candidate = _loads(text.strip())
        except ValueError:
            candidate = None
        if isinstance(candidate, dict):
            if "tool_calls" in candidate:
                raw_calls, content = candidate["tool_calls"], ""
            elif "name" in candidate and ("arguments" in candidate or "parameters" in candidate):
                raw_calls, content = [candidate], ""
            elif candidate.get("type") == "function" and "function" in candidate:
                raw_calls, content = [candidate], ""
    if not isinstance(raw_calls, list) or len(raw_calls) > 128:
        raise ValueError("模型工具调用列表无效")
    calls = []
    for call in raw_calls:
        if not isinstance(call, dict):
            raise ValueError("模型工具调用需要对象")
        function = call.get("function", call)
        if not isinstance(function, dict):
            raise ValueError("模型工具函数无效")
        name = _name(function.get("name"))
        if name not in functions:
            raise ValueError(f"模型请求了未知工具：{name}")
        args = _arguments(function.get("arguments", function.get("parameters")))
        schema = functions[name]["parameters"]
        _validate(args, schema, schema)
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, allow_nan=False)}})
    if calls and choice == "none":
        raise ValueError("tool_choice none 时模型仍生成了工具调用")
    if choice not in ("auto", "none") and not calls:
        raise ValueError("模型未生成请求要求的工具调用")
    if choice not in ("auto", "none", "required") and any(call["function"]["name"] != choice for call in calls):
        raise ValueError("模型未使用 tool_choice 指定的函数")
    if not parallel and len(calls) > 1:
        raise ValueError("parallel_tool_calls false 时模型生成了多个调用")
    return content, calls


class ToolOutputBuffer:
    """Bounded whole-completion validation for streaming (no unverified deltas)."""
    def __init__(self, tools, choice="auto", parallel=True, max_chars=2_000_000):
        self.tools, self.choice, self.parallel = tools, choice, parallel
        self.max_chars, self.parts, self.size = max_chars, [], 0
        self.reasoning = ""

    def feed(self, text):
        if not isinstance(text, str):
            raise ValueError("工具生成分块需要文字")
        self.size += len(text)
        if self.size > self.max_chars:
            raise ValueError("工具生成内容过长")
        self.parts.append(text)

    def finish(self):
        raw = "".join(self.parts)
        boundary, thinking = len(raw), False
        # Only natural-language prefixes carry thought markers. A marker inside
        # a JSON/XML argument is literal tool data and must remain byte-for-byte.
        for match in re.finditer(r"<think>|</think>|<tool_call>|\{", raw):
            tag = match.group()
            if tag == "<think>":
                thinking = True
            elif tag == "</think>":
                thinking = False
            elif not thinking:
                boundary = match.start()
                break
        visible, reasoning = [], []
        def collect(delta):
            visible.append(delta.get("content", ""))
            reasoning.append(delta.get("reasoning_content", ""))
        stream = ThinkingStream(collect)
        stream.feed(raw[:boundary])
        stream.finish()
        self.reasoning = "".join(reasoning)
        return parse_tool_output("".join(visible) + raw[boundary:], self.tools, self.choice, self.parallel)
