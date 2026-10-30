"""Personal, closed-book local-neural growth evaluation; never import test answers."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
import platform
import re
import time
import unicodedata
from pathlib import Path

import psutil
import torch
from safetensors.torch import load_file

from .benchmark import canonical, model_version, _sha_file, _BudgetStop, MAX_SECONDS
from .coldpaging import full_backbone
from .generation_checks import validate_generation_check, evaluate_generation
from .retention import eligible_learning_sample, messages, provenance

SUITE_ID = "growth-v1"
SYSTEM_PROMPT = ("Answer each task directly without references or assistance. For arithmetic tasks, "
                 "output only the number. For uppercase tasks, output only the transformed text. "
                 "For JSON extraction tasks, output only the requested JSON value. "
                 "Do not explain your answer.")
POLICY = {"temperature": 0, "seed": 42, "max_output_tokens": 64, "route": "local_neural",
          "teacher_assistance": False, "reference": False, "thinking": "off", "shots": 0,
          "system_prompt": SYSTEM_PROMPT}
POLICY_HASH = hashlib.sha256(canonical(POLICY).encode()).hexdigest()
SKILLS = {"arithmetic_add": "mathematics", "arithmetic_subtract": "mathematics",
          "arithmetic_multiply": "mathematics", "uppercase": "programming", "json_extract": "data"}
MAX_ORIGINAL_CASES = 5


def _reject_constant(_):
    raise ValueError("nonfinite JSON")


def _json_value(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=_reject_constant)
    canonical(value)
    return value


def _recognized(prompt, answer, check):
    """Whole-task grammars, not subject keywords or teacher assertions."""
    text = unicodedata.normalize("NFKC", prompt).strip()
    expression = re.fullmatch(
        r"(?:(?:Compute|Calculate|What is)\s*:?\s*|(?:计算|求)\s*:?)?"
        r"([+-]?\d{1,6})\s*([+*×−-]|加|减|乘以|乘)\s*([+-]?\d{1,6})"
        r"(?:\s*(?:等于多少|是多少))?\s*[?？。.]?\s*"
        r"(?:(?:Output|Answer|Return) only (?:the )?(?:integer|number)\.?|"
        r"[,，]?\s*(?:只输出|只返回)(?:整数|数字|结果)[。.]?)?", text, re.IGNORECASE)
    if expression:
        left, op, right = expression.groups()
        left, right = int(left), int(right)
        if op in {"+", "加"}:
            skill, expected = "arithmetic_add", left + right
        elif op in {"-", "−", "减"}:
            skill, expected = "arithmetic_subtract", left - right
        else:
            skill, expected = "arithmetic_multiply", left * right
        if re.fullmatch(r"[+-]?\d+", answer.strip()) and int(answer) == expected:
            return skill
        return None
    upper = re.fullmatch(
        r"(?:Convert\s+[\"']?([A-Za-z0-9_-]{1,64})[\"']?\s+to uppercase|"
        r"(?:把|将)(?:字符串)?\s*[\"']?([A-Za-z0-9_-]{1,64})[\"']?\s*(?:转成|转换为|转换成|变成|转换)?大写)"
        r"[。.]?\s*(?:(?:Output|Return) only (?:the )?(?:transformed text|result)\.?|"
        r"[,，]?\s*只输出(?:转换结果|结果)[。.]?)?", text, re.IGNORECASE)
    if upper:
        expected = next(value for value in upper.groups() if value is not None).upper()
        return "uppercase" if answer.strip() == expected else None
    if (check.get("validator") == "python_upper" and re.fullmatch(
            r"Write a Python function (?:that (?:takes|accepts) (?:a |one )?string and returns "
            r"(?:it|the string) in uppercase|to (?:convert a string to uppercase|uppercase a string))\.?",
            text, re.IGNORECASE) and evaluate_generation(answer, check, True)["passed"]):
        return "uppercase"
    extraction = re.fullmatch(
        r"(?:Extract|Return) (?:the )?value (?:of|for) (?:the )?key [\"']([A-Za-z0-9_-]{1,64})[\"'] "
        r"from (?:this )?JSON\s*:?\s*(\{[\s\S]*\})\s*[.]?\s*"
        r"(?:(?:Output|Return) only (?:the requested )?JSON(?: value)?\.?)?", text, re.IGNORECASE)
    chinese = re.fullmatch(r"从\s*(?:JSON\s*:?)?\s*(\{[\s\S]*\})\s*中提取(?:字段|键)\s*"
                           r"[\"']([A-Za-z0-9_-]{1,64})[\"']\s*[。.]?\s*"
                           r"(?:[,，]?\s*只输出JSON(?:值|结果)?[。.]?)?", text, re.IGNORECASE)
    if extraction or chinese:
        key, payload = extraction.groups() if extraction else tuple(reversed(chinese.groups()))
        try:
            data, gold = _json_value(payload), _json_value(answer)
            if isinstance(data, dict) and key in data and canonical(data[key]) == canonical(gold):
                return "json_extract"
        except (ValueError, TypeError, RecursionError):
            pass
    return None


def eligible_cases(runtime):
    """Private source cases; independent original-gold checks precede any skill label."""
    output = []
    for sample in runtime.memory.samples():
        try:
            if (sample.get("approved") not in (True, 1) or sample.get("training_allowed") not in (True, 1)
                    or not eligible_learning_sample(sample)):
                continue
            context = messages(sample)
            if context[-1]["role"] != "user":
                continue
            answer = sample.get("answer")
            if (not isinstance(answer, str) or not answer.strip() or len(answer) > 8192
                    or len(context[-1]["content"]) > 4096 or not isinstance(sample.get("id"), str)):
                continue
            check = validate_generation_check(provenance(sample).get("generation_check"), default_expected=answer)
            skill = _recognized(context[-1]["content"], answer, check)
            if skill is None or not evaluate_generation(answer, check, True)["passed"]:
                continue
            output.append({"source_id": sample["id"], "domain": SKILLS[skill], "skill": skill,
                           "prompt": context[-1]["content"], "messages": copy.deepcopy(context),
                           "answer": answer, "generation_check": check})
        except (ValueError, TypeError, KeyError, RecursionError):
            continue
    return output


def eligible_skills(runtime):
    return sorted({case["skill"] for case in eligible_cases(runtime)})


def _read_json(path):
    _sha_file(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("个人评测神经版本记录无效")
    return value


def _verify_live(module, weights):
    """Compare selected immutable tensors with actual resident tensors, not file names."""
    saved = load_file(str(weights))
    actual = module.state_dict()
    if saved.keys() != actual.keys():
        raise ValueError("个人评测内存权重与文件结构不一致")
    for key, value in actual.items():
        if (value.is_meta or not torch.isfinite(value).all() or saved[key].shape != value.shape
                or not torch.equal(saved[key].to(dtype=value.dtype), value.detach().cpu())):
            raise ValueError("个人评测内存权重与已保存版本不一致")


def personal_model_version(runtime):
    """Bind core plus selected learned branch and neural router, including gate metadata."""
    with runtime.lock:
        base_core = model_version(runtime)
        store = Path(runtime.store)
        branch = getattr(runtime, "branch", None)
        branch_binding = None
        pointer = store / "active.json"
        if pointer.exists() or pointer.is_symlink():
            active = _read_json(pointer)
            expert_id = active.get("id")
            if (branch is None or not isinstance(expert_id, str)
                    or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", expert_id)
                    or branch.expert_id != expert_id or active.get("identity") != runtime.identity
                    or active.get("width") != branch.width):
                raise ValueError("个人评测活动专家与内存版本不一致")
            weights, metadata = branch._paths(expert_id)
            record = _read_json(metadata)
            digest = _sha_file(weights)
            if (record.get("identity") != branch.identity
                    or record.get("sha256") != digest
                    or record.get("width") != branch.width or record.get("hidden") != branch.hidden
                    or record.get("core_version") != active.get("core_version")
                    or getattr(runtime, "branch_core_version", None) != active.get("core_version")):
                raise ValueError("个人评测专家文件校验失败")
            if branch.expert is not None:
                _verify_live(branch.expert, weights)
            branch_binding = {"pointer_sha256": _sha_file(pointer), "weights_sha256": digest,
                              "metadata_sha256": _sha_file(metadata), "id": expert_id,
                              "requires_gate": branch.requires_gate}
        elif branch is not None and (branch.expert is not None or branch.expert_id is not None):
            raise ValueError("个人评测不能使用未提交的专家")
        router_binding = None
        router_meta = getattr(runtime, "router_meta", {"calibrated": False})
        router_pointer = store / "active-router.json"
        if router_pointer.exists() or router_pointer.is_symlink():
            active = _read_json(router_pointer)
            version = active.get("id")
            if (not isinstance(version, str) or not re.fullmatch(r"router-[a-f0-9]{12}", version)
                    or active.get("identity") != runtime.identity):
                raise ValueError("个人评测快脑活动版本无效")
            weights, metadata = store / (version + ".safetensors"), store / (version + ".json")
            record = _read_json(metadata)
            if (record.get("id") != version or record.get("core_version") != active.get("core_version")
                    or active.get("sha256") != _sha_file(weights)
                    or active.get("metadata_sha256") != _sha_file(metadata)):
                raise ValueError("个人评测快脑指针校验失败")
            router_binding = {"pointer_sha256": _sha_file(router_pointer)}
        else:
            weights, metadata = store / "router.safetensors", store / "router.json"
            if router_meta.get("calibrated") or router_meta.get("gates_expert"):
                record = _read_json(metadata)
                router_binding = {"legacy": True}
        if router_binding is not None:
            if record.get("identity") != runtime.identity or record.get("sha256") != _sha_file(weights):
                raise ValueError("个人评测快脑权重校验失败")
            effective = record if record.get("core_version") == runtime.active_core else {**record, "calibrated": False}
            if canonical(effective) != canonical(router_meta) or getattr(runtime, "router", None) is None:
                raise ValueError("个人评测快脑内存元数据不一致")
            _verify_live(runtime.router, weights)
            router_binding.update(weights_sha256=_sha_file(weights), metadata_sha256=_sha_file(metadata))
        elif router_meta != {"calibrated": False, "samples": 0} and router_meta != {"calibrated": False}:
            raise ValueError("个人评测快脑缺少可核验的版本")
        return hashlib.sha256(canonical({"base_core": base_core, "branch": branch_binding,
            "router": router_binding, "router_meta": router_meta, "route": "local_neural"}).encode()).hexdigest()


def _challenge(challenge):
    if not isinstance(challenge, dict) or challenge.get("suite_id") != SUITE_ID:
        raise ValueError("成长题库版本不兼容")
    if canonical(challenge.get("policy")) != canonical(POLICY) or challenge.get("policy_hash") != POLICY_HASH:
        raise ValueError("成长评测条件不一致")
    for key in ("model_version", "suite_sha256"):
        if not isinstance(challenge.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", challenge[key]):
            raise ValueError("成长评测摘要无效")
    if not isinstance(challenge.get("challenge_id"), str) or not 1 <= len(challenge["challenge_id"]) <= 128:
        raise ValueError("成长挑战无效")
    domain, skills = challenge.get("domain"), challenge.get("skills")
    if (domain not in set(SKILLS.values()) or not isinstance(skills, list) or not skills
            or any(not isinstance(skill, str) or skill not in SKILLS or SKILLS[skill] != domain for skill in skills)
            or len(skills) != len(set(skills))):
        raise ValueError("成长评测领域或技能无效")
    questions, ids = challenge.get("questions"), set()
    if not isinstance(questions, list) or len(questions) != 24:
        raise ValueError("成长题目不完整")
    for question in questions:
        if (not isinstance(question, dict) or set(question) != {"id", "skill", "prompt"}
                or not isinstance(question.get("id"), str)
                or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", question["id"])
                or question["id"] in ids or question.get("skill") not in skills
                or not isinstance(question.get("prompt"), str) or not 1 <= len(question["prompt"]) <= 4096):
            raise ValueError("成长题目格式无效")
        ids.add(question["id"])
    return questions


def _run(runtime, questions, expected, stop, progress, *, force_local=False):
    with runtime.lock:
        stop.check()
        if (runtime.state not in {"ready", "evaluating"} or runtime.model is None
                or runtime.tokenizer is None or getattr(runtime, "training_session", None) is not None):
            raise RuntimeError("模型尚未就绪或正在学习，不能测成长")
        runtime._ready()
        if personal_model_version(runtime) != expected:
            raise ValueError("个人神经版本已变化，请重新测成长")
        state = runtime.state
        training_flags = [(module, module.training) for module in runtime.model.modules()]
        router = getattr(runtime, "router", None)
        router_training = router.training if router is not None else None
        branch = getattr(runtime, "branch", None)
        original_expert = branch.expert if branch else None
        original_hint = runtime.__dict__.get("external_hint", None)
        had_hint = "external_hint" in runtime.__dict__
        started = time.perf_counter()
        answers, timings = [], []
        runtime.state = "evaluating"
        runtime.external_hint = lambda _: None
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                runtime.model.eval()
                if router is not None:
                    router.eval()
                for index, question in enumerate(questions):
                    stop.check()
                    if branch:
                        branch.expert = original_expert
                    if personal_model_version(runtime) != expected:
                        raise ValueError("成长评测中个人神经版本已变化")
                    tick = time.perf_counter()
                    context = question.get("messages") or [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question["prompt"]}]
                    route = runtime.choose_route(context)
                    if force_local and (route.get("route") == "teacher" or route.get("action") == "teacher"):
                        route = {"route": "local", "action": "local_core"}
                    stop.check()
                    if route.get("route") == "teacher" or route.get("action") == "teacher":
                        answer = {"id": question["id"], "text": "", "finish_reason": "abstain"}
                        prompt_count = completion_count = 0
                        actual_action = "abstain"
                        branch_enabled = False
                    else:
                        if (route.get("route") != "local" or route.get("action") not in {"local_core", "neural_expert"}):
                            raise ValueError("成长评测路由无效")
                        gated = branch and (branch.requires_gate or runtime.router_meta.get("gates_expert"))
                        use_expert = branch is not None and (not gated or route["action"] == "neural_expert")
                        use_expert = use_expert and (branch.expert is not None or branch.expert_id is not None)
                        with full_backbone(runtime) if use_expert else nullcontext():
                            if use_expert:
                                runtime._load_active_expert()
                            elif branch:
                                branch.expert = None
                            if route["action"] == "neural_expert" and (branch is None or branch.expert is None):
                                raise ValueError("成长评测选择的专家未加载")
                            tokens = runtime._tokens(context, "off", output_reserve=POLICY["max_output_tokens"])
                            output = runtime.generate_tokens(tokens, POLICY["max_output_tokens"], temperature=0.0, stop=stop)
                            branch_enabled = branch is not None and branch.expert is not None
                            actual_action = "neural_expert" if branch_enabled else "local_core"
                        stop.check()
                        generated = output[0, tokens.shape[1]:]
                        text = runtime.tokenizer.decode(generated, skip_special_tokens=True).strip()
                        if not text or len(text) > 8192:
                            raise ValueError("成长评测回复为空或过长")
                        eos = getattr(getattr(runtime.model, "generation_config", None), "eos_token_id", None)
                        eos = runtime.tokenizer.eos_token_id if eos is None else eos
                        eos_ids = eos if isinstance(eos, list) else [eos]
                        complete = generated.numel() > 0 and int(generated[-1]) in eos_ids
                        answer = {"id": question["id"], "text": text,
                                  "finish_reason": "stop" if complete else "length"}
                        prompt_count, completion_count = int(tokens.shape[1]), int(generated.numel())
                    answers.append(answer)
                    timings.append({"id": question["id"], "seconds": time.perf_counter() - tick,
                                    "prompt_tokens": prompt_count, "completion_tokens": completion_count,
                                    "local_action": actual_action, "branch_enabled": branch_enabled})
                    progress({"completed": index + 1, "total": len(questions)})
                if branch:
                    branch.expert = original_expert
                stop.check()
                if personal_model_version(runtime) != expected:
                    raise InterruptedError("成长评测已停止或个人版本已变化")
            stop.check()
            return answers, {"elapsed_seconds": time.perf_counter() - started, "items": timings,
                "device": str(runtime.device), "precision": str(runtime.dtype), "platform": platform.platform(),
                "torch_version": torch.__version__, "rss_after_bytes": psutil.Process().memory_info().rss,
                "inference": runtime.inference_identity(),
                "teacher_calls": 0, "reference": False, "route": "local_neural",
                "branch_enabled": any(item["branch_enabled"] for item in timings),
                "time_budget_seconds": MAX_SECONDS,
                "timeout_boundary": "checked per generated token; cannot preempt an in-flight kernel or weight load"}
        finally:
            if branch:
                branch.expert = original_expert
            if had_hint:
                runtime.external_hint = original_hint
            else:
                runtime.__dict__.pop("external_hint", None)
            for module, training in training_flags:
                module.training = training
            if router is not None:
                router.training = router_training
            runtime.state = state


def run_growth(runtime, challenge, stop=None, progress=None, *, deadline_monotonic=None):
    stop = _BudgetStop(stop, deadline_monotonic)
    stop.check()
    questions = _challenge(challenge)
    answers, metadata = _run(runtime, questions, challenge["model_version"],
        stop, progress or (lambda _: None))
    return {"challenge_id": challenge["challenge_id"], "model_version": challenge["model_version"],
            "answers": answers, "metadata": {**metadata, "suite_id": SUITE_ID,
            "suite_sha256": challenge["suite_sha256"], "policy_hash": POLICY_HASH}}


def original_retest(runtime, cases, stop=None, *, deadline_monotonic=None):
    """Private, seen-original diagnostics; never an input to the public growth score."""
    stop = _BudgetStop(stop, deadline_monotonic)
    stop.check()
    valid = eligible_cases(runtime)
    by_id = {case["source_id"]: case for case in valid}
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, dict) or case != by_id.get(case.get("source_id")) for case in cases)
            or len({case["source_id"] for case in cases}) != len(cases)):
        raise ValueError("原题复测只允许当前合格且已核对的私有材料")
    available = len(cases)
    cases = cases[:MAX_ORIGINAL_CASES]
    questions = [{"id": str(index), "messages": case["messages"], "prompt": case["prompt"]}
                 for index, case in enumerate(cases)]
    version = personal_model_version(runtime)
    answers, metadata = _run(runtime, questions, version,
        stop, lambda _: None)
    results = [{"source_id": case["source_id"], "domain": case["domain"], "skill": case["skill"],
                "answer": answer["text"], "finish_reason": answer["finish_reason"],
                "passed": evaluate_generation(answer["text"], case["generation_check"],
                    answer["finish_reason"] == "stop")["passed"]} for case, answer in zip(cases, answers)]
    stop.check()
    return {"model_version": version, "cases": results,
            "metadata": {**metadata, "available_original_cases": available, "tested_original_cases": len(cases),
                         "original_case_limit": MAX_ORIGINAL_CASES},
            "private": True, "counts_toward_public_growth": False}
