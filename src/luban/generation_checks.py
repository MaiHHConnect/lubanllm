"""Finite generation checks; parse code as AST and never execute model output."""
import ast
import json
import re
import unicodedata

from .goal_topics import REQUIRED as GOAL_TOPICS, evaluate_goal_topics

FINITE_VALIDATORS = {"integer", "json", "python_upper", "translation", "normalized_exact", "one_of", "goal_topics"}
VALIDATORS = FINITE_VALIDATORS | {"nonempty"}


def is_finite_generation_check(check):
    check = validate_generation_check(check)
    return check["type"] == "exact" or (check["type"] == "semantic" and check["validator"] in FINITE_VALIDATORS)


def validate_generation_check(check, *, default_expected=None):
    if check is None:
        check = {"type": "exact", "expected": default_expected}
    if (not isinstance(check, dict) or not isinstance(check.get("type"), str) or
            check["type"] not in {"exact", "contains", "semantic"}):
        raise ValueError("生成检验类型无效")
    check = dict(check)
    maximum = check.get("max_new_tokens", 192)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 512:
        raise ValueError("核心生成检验预算须在 1–512 token 内")
    forbidden = check.get("forbidden", [])
    if (not isinstance(forbidden, list) or len(forbidden) > 32 or
            any(not isinstance(fragment, str) or not fragment.strip() or len(fragment) > 512 for fragment in forbidden)):
        raise ValueError("核心生成禁用片段无效")
    check.update(max_new_tokens=maximum, forbidden=list(forbidden))
    if check["type"] in {"exact", "contains"}:
        expected = check.get("expected")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("生成检验 expected 必须为非空字符串")
        check["expected"] = expected.strip()
        return check
    validator = check.get("validator")
    if not isinstance(validator, str) or validator not in VALIDATORS:
        raise ValueError("语义检验 validator 无效")
    expected = check.get("expected")
    if validator == "goal_topics" and (not isinstance(expected, str) or expected not in GOAL_TOPICS):
        raise ValueError("目标说明检验需要预先冻结的有效 topic")
    if validator == "normalized_exact" and (not isinstance(expected, str) or not expected.strip()):
        raise ValueError("规范化精确检验需要非空字符串 expected")
    if validator == "integer" and (isinstance(expected, bool) or not isinstance(expected, int)):
        raise ValueError("整数检验需要整数 expected")
    if validator == "json":
        try:
            json.dumps(expected, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("JSON 检验 expected 无效") from error
        if "expected" not in check:
            raise ValueError("JSON 检验需要 expected")
    if validator in {"translation", "one_of"}:
        accepted = check.get("accepted")
        if (not isinstance(accepted, list) or not 1 <= len(accepted) <= 16 or
                any(not isinstance(value, str) or not value.strip() or len(value) > 256 for value in accepted)):
            raise ValueError("翻译检验需要明确的 accepted 译文" if validator == "translation" else "候选精确检验需要明确的 accepted 文本")
        check["accepted"] = list(accepted)
    return check


def _unfence(answer):
    match = re.fullmatch(r"```(?:python|py|json)?[ \t]*\n([\s\S]*?)\n```", answer, flags=re.IGNORECASE)
    return match[1] if match else answer


def _python_upper(answer):
    """A strict functional subset: one single-argument def returning str.upper()."""
    try:
        module = ast.parse(_unfence(answer))
    except (SyntaxError, ValueError, RecursionError):
        return False
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return False
    function = module.body[0]
    arguments = function.args.posonlyargs + function.args.args
    if (len(arguments) != 1 or function.decorator_list or function.args.vararg or function.args.kwarg or
            function.args.kwonlyargs or function.args.defaults):
        return False
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "upper"
            and isinstance(value.func.value, ast.Name) and value.func.value.id == arguments[0].arg
            and not value.args and not value.keywords):
        return False
    # Reject calls hidden in annotations or defaults, even though nothing is executed.
    return all(not isinstance(node, ast.Call) or node is value for node in ast.walk(function))


def _json(answer, expected):
    def unique_pairs(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = value
        return output

    try:
        observed = json.loads(_unfence(answer), object_pairs_hook=unique_pairs)
        return (json.dumps(observed, sort_keys=True, allow_nan=False, ensure_ascii=False) ==
                json.dumps(expected, sort_keys=True, allow_nan=False, ensure_ascii=False))
    except (ValueError, TypeError, RecursionError):
        return False


def evaluate_generation(answer, check, complete):
    check = validate_generation_check(check)
    if not isinstance(answer, str) or not isinstance(complete, bool):
        raise ValueError("生成文本与结束标记无效")
    answer = answer.strip()
    kind, expected = check["type"], check.get("expected")
    validator = check.get("validator")
    if kind == "exact":
        correct = answer == expected
    elif kind == "contains":
        correct = expected in answer
    elif validator == "integer":
        try:
            correct = bool(re.fullmatch(r"[+-]?\d+", answer)) and int(answer) == expected
        except ValueError:
            correct = False
    elif validator == "json":
        correct = _json(answer, expected)
    elif validator == "goal_topics":
        correct = evaluate_goal_topics(answer, expected, complete)
    elif validator == "normalized_exact":
        correct = unicodedata.normalize("NFKC", answer) == unicodedata.normalize("NFKC", expected.strip())
    elif validator == "python_upper":
        correct = _python_upper(answer)
    elif validator == "one_of":
        correct = answer in check["accepted"]
    elif validator == "translation":
        normalized = answer.strip(" \t\r\n\"'“”‘’。.!！")
        correct = any(normalized.casefold() == value.strip().casefold() for value in check["accepted"])
    else:
        correct = bool(answer)
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    found = [fragment for fragment in check["forbidden"] if unicodedata.normalize("NFKC", fragment).casefold() in normalized]
    return {"passed": bool(complete and correct and not found), "correct": bool(correct), "complete": complete,
            "forbidden_found": found, "check": kind, "validator": validator,
            "expected": check.get("accepted", expected)}
