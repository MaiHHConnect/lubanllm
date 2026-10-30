import copy
import hashlib
import json
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from luban.decision import FastDecisionHead
from luban.growth import (POLICY, POLICY_HASH, SUITE_ID, eligible_cases, eligible_skills,
                          original_retest, personal_model_version, run_growth)
from luban.memory import Memory
from luban.runtime import Runtime
from luban.benchmark import _StopRequested
from luban.coldpaging import full_backbone


class Branch(nn.Module):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.base = nn.Linear(2, 2)
        self.expert = None
        self.expert_id = None
        self.width = self.hidden = 2
        self.requires_gate = False

    def _paths(self, identifier):
        return self.store / (identifier + ".safetensors"), self.store / (identifier + ".json")


class Generator(nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self.branch = runtime.branch
        self.calls = []
        self.mode = "normal"

    def generate(self, tokens, **kwargs):
        self.calls.append({**kwargs, "expert_present": self.branch.expert is not None})
        assert not kwargs["do_sample"]
        assert kwargs["max_new_tokens"] == 64
        if self.mode == "error":
            raise RuntimeError("generator failed")
        if self.mode == "cancel":
            kwargs["stopping_criteria"][0].event.set()
        if self.mode == "change":
            self.runtime.identity = "e" * 64
        values = [] if self.mode == "empty" else [5] * 64 if self.mode == "length" else [5, 9]
        return torch.cat((tokens, torch.tensor([values], dtype=torch.long)), dim=1)


class Tokenizer:
    eos_token_id = 2
    def decode(self, tokens, skip_special_tokens):
        return "7" if tokens.numel() else ""


class GrowthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        memory = Memory(self.store / "memory")
        self.runtime = SimpleNamespace(lock=threading.RLock(), state="ready", identity="d" * 64,
            active_core=None, store=self.store, memory=memory, training_session=None, device="cpu",
            dtype=torch.float32, cold_backbone=None, tokenizer=Tokenizer(),
            router_meta={"calibrated": False, "samples": 0}, router=FastDecisionHead(2))
        self.runtime.branch = Branch(self.store)
        self.runtime.branch.identity = self.runtime.identity + ":layer-23"
        self.runtime.model = Generator(self.runtime)
        self.runtime.model.generation_config = SimpleNamespace(eos_token_id=[2, 9])
        self.runtime.inference_identity = lambda: {"engine": "hf"}
        def generate_tokens(tokens, maximum, *, temperature=0.0, emit=None, stop=None):
            with full_backbone(self.runtime):
                return self.runtime.model.generate(tokens, attention_mask=torch.ones_like(tokens),
                    max_new_tokens=maximum, do_sample=temperature > 0,
                    pad_token_id=self.runtime.tokenizer.eos_token_id,
                    stopping_criteria=[_StopRequested(stop)])
        self.runtime.generate_tokens = generate_tokens
        self.runtime._ready = lambda: None
        self.token_calls = []
        def tokens(messages, thinking="off", **kwargs):
            self.token_calls.append((messages, thinking, kwargs))
            return torch.tensor([[1, 3]])
        self.runtime._tokens = tokens
        self.runtime.shallow = lambda _: torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
        self.runtime.external_hint = lambda _: (_ for _ in ()).throw(AssertionError("reference consulted"))
        self.runtime.choose_route = types.MethodType(Runtime.choose_route, self.runtime)
        self.runtime._load_active_expert = self.load_expert
        self.challenge = {"suite_id": SUITE_ID, "challenge_id": "private-test", "domain": "mathematics",
            "skills": ["arithmetic_add"], "policy": copy.deepcopy(POLICY), "policy_hash": POLICY_HASH,
            "suite_sha256": "f" * 64, "model_version": personal_model_version(self.runtime),
            "questions": [{"id": f"growth-{i}", "skill": "arithmetic_add", "prompt": "Compute 3 + 4."}
                          for i in range(24)]}

    def load_expert(self):
        if self.runtime.branch.expert is None and self.runtime.branch.expert_id:
            weights, _ = self.runtime.branch._paths(self.runtime.branch.expert_id)
            from safetensors.torch import load_file
            module = nn.Linear(2, 2)
            module.load_state_dict(load_file(str(weights)))
            self.runtime.branch.expert = module

    def save_branch(self, identifier="branch-one", value=1.0, requires_gate=False):
        branch = self.runtime.branch
        module = nn.Linear(2, 2)
        with torch.no_grad():
            module.weight.fill_(value)
            module.bias.fill_(0)
        branch.expert, branch.expert_id = module, identifier
        branch.requires_gate = requires_gate
        self.runtime.branch_core_version = None
        weights, metadata = branch._paths(identifier)
        save_file(module.state_dict(), str(weights))
        record = {"identity": branch.identity, "hidden": 2, "width": 2,
                  "core_version": None, "requires_gate": requires_gate,
                  "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        metadata.write_text(json.dumps(record))
        (self.store / "active.json").write_text(json.dumps({"id": identifier, "identity": self.runtime.identity,
                                                           "width": 2, "core_version": None}))

    def save_router(self, action="neural_expert"):
        router = self.runtime.router
        with torch.no_grad():
            router.projection.weight.zero_()
            router.projection.bias.zero_()
            router.projection.bias[FastDecisionHead.actions.index(action)] = 10
        version = "router-0123456789ab"
        weights, metadata = self.store/(version+".safetensors"), self.store/(version+".json")
        save_file(router.state_dict(), str(weights))
        meta = {"id": version, "identity": self.runtime.identity, "core_version": None,
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "calibrated": True,
                "supported_actions": list(FastDecisionHead.actions), "temperature": 1.0, "threshold": 0.8,
                # Test fixture satisfies the production expert-scope contract; the real
                # calibration path supplies a measured centroid and non-negative threshold.
                "expert_scope": {"required": True, "passed": True,
                                  "centroid": [1.0] * router.projection.in_features,
                                  "threshold": -1.0}}
        metadata.write_text(json.dumps(meta))
        (self.store/"active-router.json").write_text(json.dumps({"id": version,
            "identity": self.runtime.identity, "core_version": None, "sha256": meta["sha256"],
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()}))
        self.runtime.router_meta = meta

    def source(self, prompt, answer, **kwargs):
        return self.runtime.memory.add_sample(prompt, answer, approved=True, training_allowed=True, **kwargs)

    def test_precise_skill_matchers_independently_verify_gold(self):
        self.source("Compute 3 + 4. Output only the number.", "7")
        self.source("计算12-5，只返回整数。", "7")
        self.source("Calculate 6 * 8.", "48")
        self.source('Convert "signal7" to uppercase. Output only the transformed text.', "SIGNAL7")
        self.source('Extract the value of key "n" from JSON {"n":7,"x":2}. Output only JSON.', "7")
        self.assertEqual(eligible_skills(self.runtime), ["arithmetic_add", "arithmetic_multiply",
                         "arithmetic_subtract", "json_extract", "uppercase"])
        cases = eligible_cases(self.runtime)
        self.assertEqual(len(cases), 5)
        self.assertEqual({case["domain"] for case in cases}, {"mathematics", "programming", "data"})
        self.assertTrue(all(case["messages"] and "generation_check" in case for case in cases))

    def test_question_subject_keywords_do_not_assign_math_or_programming(self):
        for prompt, answer in [("My pet is called Maths. What is its name?", "Maths"),
                               ("Explain why 3 + 4 appears in this poem.", "7"),
                               ("What is my favorite programming language?", "Python"),
                               ("Compute 3 + 4.", "8"),
                               ('Convert "abc" to uppercase.', "abc"),
                               ('Extract value of key "n" from JSON {"n":7}.', "8")]:
            self.source(prompt, answer)
        self.assertEqual(eligible_cases(self.runtime), [])

    def test_teacher_opinion_is_not_verification(self):
        self.source("Compute 3 + 4.", "7", source="teacher", provenance={"teacher_verified": True})
        self.assertEqual(eligible_skills(self.runtime), [])
        self.source("Compute 4 + 3.", "7", source="teacher", provenance={"program_verified": True,
            "verification_evidence": "independent integer program checked"})
        self.assertEqual(eligible_skills(self.runtime), ["arithmetic_add"])

    def test_permissions_and_route_only_records_are_excluded(self):
        identifier = self.source("Compute 3 + 4.", "7")
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET training_allowed=0 WHERE id=?", (identifier,))
        self.source("Compute 5 + 2.", "7", provenance={"route_only": True})
        self.source("Compute 6 + 1.", "7", provenance={"generation_check": {"type": "exact", "expected": "8"}})
        self.assertEqual(eligible_cases(self.runtime), [])

    def test_program_answer_is_only_parsed_never_executed(self):
        self.source("Write a Python function to uppercase a string.", "def up(s):\n    return s.upper()",
                    provenance={"generation_check": {"type": "semantic", "validator": "python_upper"}})
        self.assertEqual(eligible_skills(self.runtime), ["uppercase"])

    def test_json_duplicate_keys_and_wrong_types_do_not_qualify(self):
        self.source('Extract the value of key "x" from JSON {"x":1,"x":2}.', "2")
        self.source('Extract the value of key "x" from JSON {"x":true}.', "1")
        self.assertEqual(eligible_skills(self.runtime), [])

    def test_complete_messages_are_private_and_preserved(self):
        context = [{"role": "system", "content": "Answer directly."},
                   {"role": "user", "content": "Compute 3 + 4."}]
        self.source("different display label", "7", provenance={"messages": context})
        cases = eligible_cases(self.runtime)
        self.assertEqual(cases[0]["messages"], context)
        cases[0]["messages"][0]["content"] = "modified copy"
        self.assertEqual(eligible_cases(self.runtime)[0]["messages"], context)

    def test_branch_and_router_are_bound_to_actual_files_and_live_tensors(self):
        initial = personal_model_version(self.runtime)
        self.save_branch()
        branch_version = personal_model_version(self.runtime)
        self.assertNotEqual(initial, branch_version)
        self.save_branch("branch-two", 2)
        self.assertNotEqual(branch_version, personal_model_version(self.runtime))
        self.save_router()
        router_version = personal_model_version(self.runtime)
        self.runtime.router.projection.bias.data[0] += 1
        with self.assertRaises(ValueError):
            personal_model_version(self.runtime)
        self.assertEqual(len(router_version), 64)

    def test_layer_bound_expert_metadata_is_accepted(self):
        self.save_branch()
        metadata = self.store / "branch-one.json"
        value = json.loads(metadata.read_text())
        value["identity"] = self.runtime.identity + ":layer-23"
        metadata.write_text(json.dumps(value))
        self.assertEqual(len(personal_model_version(self.runtime)), 64)
        value["identity"] = self.runtime.identity + ":layer-22"
        metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "专家文件校验失败"):
            personal_model_version(self.runtime)

    def test_unsaved_branch_and_changed_active_pointer_fail_closed(self):
        self.runtime.branch.expert = nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            personal_model_version(self.runtime)
        self.save_branch()
        active = self.store / "active.json"
        value = json.loads(active.read_text())
        value["id"] = "wrong"
        active.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            personal_model_version(self.runtime)

    def test_true_neural_route_enables_learned_expert_without_hints(self):
        self.save_branch()
        self.save_router("neural_expert")
        self.challenge["model_version"] = personal_model_version(self.runtime)
        original, hint = self.runtime.branch.expert, self.runtime.external_hint
        rng = torch.get_rng_state().clone()
        memory_before = self.runtime.memory.samples()
        progress = []
        result = run_growth(self.runtime, self.challenge, progress=progress.append)
        self.assertEqual(len(result["answers"]), 24)
        self.assertTrue(all(call["expert_present"] for call in self.runtime.model.calls))
        self.assertTrue(all(item["local_action"] == "neural_expert" for item in result["metadata"]["items"]))
        self.assertTrue(all(answer["finish_reason"] == "stop" for answer in result["answers"]))
        self.assertEqual(progress[-1], {"completed": 24, "total": 24})
        self.assertIs(self.runtime.branch.expert, original)
        self.assertIs(self.runtime.external_hint, hint)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(self.runtime.memory.samples(), memory_before)
        self.assertEqual(self.runtime.state, "ready")
        self.assertTrue(self.runtime.model.training)
        self.assertEqual(result["metadata"]["teacher_calls"], 0)
        self.assertFalse(result["metadata"]["reference"])
        self.assertNotIn("score", result)

    def test_mixed_train_flags_router_and_inherited_hint_are_restored(self):
        self.runtime.model.train()
        self.runtime.branch.base.eval()
        self.runtime.router.train()
        del self.runtime.external_hint
        before = [module.training for module in self.runtime.model.modules()]
        run_growth(self.runtime, self.challenge)
        self.assertEqual([module.training for module in self.runtime.model.modules()], before)
        self.assertTrue(self.runtime.router.training)
        self.assertNotIn("external_hint", self.runtime.__dict__)

    def test_cold_expert_is_actually_loaded_then_restored_to_cold(self):
        self.save_branch()
        self.save_router("neural_expert")
        self.runtime.branch.expert = None
        self.challenge["model_version"] = personal_model_version(self.runtime)
        class Pager:
            depth_count = 0
            def acquire(self): self.depth_count += 1
            def release(self): self.depth_count -= 1
        pager = Pager()
        self.runtime.cold_backbone = pager
        run_growth(self.runtime, self.challenge)
        self.assertTrue(all(call["expert_present"] for call in self.runtime.model.calls))
        self.assertIsNone(self.runtime.branch.expert)
        self.assertEqual(pager.depth_count, 0)

    def test_router_saved_weight_change_changes_same_core_personal_version(self):
        from luban.benchmark import model_version
        self.save_router("local_core")
        core = model_version(self.runtime)
        first = personal_model_version(self.runtime)
        self.save_router("teacher")
        self.assertEqual(core, model_version(self.runtime))
        self.assertNotEqual(first, personal_model_version(self.runtime))

    def test_teacher_route_abstains_without_generation_or_paid_call(self):
        self.save_router("teacher")
        self.challenge["model_version"] = personal_model_version(self.runtime)
        result = run_growth(self.runtime, self.challenge)
        self.assertTrue(all(row["finish_reason"] == "abstain" and row["text"] == "" for row in result["answers"]))
        self.assertEqual(self.runtime.model.calls, [])

    def test_core_route_does_not_leave_expert_enabled(self):
        self.save_branch(requires_gate=True)
        self.save_router("local_core")
        self.challenge["model_version"] = personal_model_version(self.runtime)
        original = self.runtime.branch.expert
        run_growth(self.runtime, self.challenge)
        self.assertTrue(all(not call["expert_present"] for call in self.runtime.model.calls))
        self.assertIs(self.runtime.branch.expert, original)

    def test_legacy_ungated_expert_remains_enabled_on_default_core_route(self):
        self.save_branch()
        self.challenge["model_version"] = personal_model_version(self.runtime)
        result = run_growth(self.runtime, self.challenge)
        self.assertTrue(all(call["expert_present"] for call in self.runtime.model.calls))
        self.assertTrue(result["metadata"]["branch_enabled"])
        self.assertTrue(all(item["branch_enabled"] and item["local_action"] == "neural_expert"
                            for item in result["metadata"]["items"]))

    def test_gated_expert_runs_only_for_actual_neural_route(self):
        self.save_branch(requires_gate=True)
        self.save_router("neural_expert")
        self.challenge["model_version"] = personal_model_version(self.runtime)
        result = run_growth(self.runtime, self.challenge)
        self.assertTrue(all(call["expert_present"] for call in self.runtime.model.calls))
        self.assertTrue(result["metadata"]["branch_enabled"])

    def test_truncation_is_not_complete(self):
        self.runtime.model.mode = "length"
        result = run_growth(self.runtime, self.challenge)
        self.assertTrue(all(row["finish_reason"] == "length" for row in result["answers"]))

    def test_cancel_empty_error_and_version_change_restore_all_state(self):
        hint = self.runtime.external_hint
        for mode, error in (("cancel", InterruptedError), ("empty", ValueError),
                            ("error", RuntimeError), ("change", InterruptedError)):
            with self.subTest(mode=mode):
                self.runtime.identity = "d" * 64
                self.runtime.model.mode = mode
                with self.assertRaises((error, ValueError)):
                    run_growth(self.runtime, self.challenge)
                self.assertEqual(self.runtime.state, "ready")
                self.assertIs(self.runtime.external_hint, hint)
                self.assertTrue(self.runtime.model.training)

    def test_cancel_before_first_route_and_generation(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            run_growth(self.runtime, self.challenge, stop)
        self.assertEqual(self.runtime.model.calls, [])

    def test_missing_duplicate_gold_and_cross_domain_questions_are_rejected(self):
        cases = []
        for field, value in (("questions", self.challenge["questions"][:-1]),
                             ("domain", "pets"), ("skills", ["uppercase"]),
                             ("policy", {**POLICY, "reference": True})):
            cases.append({**self.challenge, field: value})
        for mutation in ("duplicate", "gold"):
            candidate = copy.deepcopy(self.challenge)
            if mutation == "duplicate": candidate["questions"][1]["id"] = candidate["questions"][0]["id"]
            else: candidate["questions"][0]["answer"] = "7"
            cases.append(candidate)
        for candidate in cases:
            with self.assertRaises(ValueError):
                run_growth(self.runtime, candidate)
        self.assertEqual(self.runtime.model.calls, [])

    def test_original_retest_is_seen_private_diagnostic_never_public_score(self):
        self.source("Compute 3 + 4.", "7")
        result = original_retest(self.runtime, eligible_cases(self.runtime))
        self.assertTrue(result["private"])
        self.assertFalse(result["counts_toward_public_growth"])
        self.assertTrue(result["cases"][0]["passed"])
        self.assertNotIn("growth", result)
        self.assertEqual(self.runtime.memory.samples()[0]["answer"], "7")

    def test_original_retest_cannot_take_forged_or_revoked_gold(self):
        identifier = self.source("Compute 3 + 4.", "7")
        cases = eligible_cases(self.runtime)
        cases[0]["answer"] = "8"
        with self.assertRaises(ValueError): original_retest(self.runtime, cases)
        cases = eligible_cases(self.runtime)
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET approved=0 WHERE id=?", (identifier,))
        with self.assertRaises(ValueError): original_retest(self.runtime, cases)

    def test_duplicate_private_original_cannot_inflate_diagnostic(self):
        self.source("Compute 3 + 4.", "7")
        cases = eligible_cases(self.runtime)
        with self.assertRaises(ValueError): original_retest(self.runtime, cases * 2)

    def test_protocol_short_answer_budget_is_64_in_generation_and_context(self):
        self.assertEqual(POLICY["max_output_tokens"], 64)
        run_growth(self.runtime, self.challenge)
        self.assertTrue(all(kwargs.get("output_reserve") == 64 for _, _, kwargs in self.token_calls))

    def test_deadline_before_first_token_cannot_produce_answers(self):
        with patch("luban.growth.time.monotonic", return_value=100):
            with self.assertRaises(TimeoutError):
                run_growth(self.runtime, self.challenge, deadline_monotonic=99)
        self.assertEqual(self.runtime.model.calls, [])
        self.assertEqual(self.runtime.state, "ready")

    def test_token_stopping_criterion_observes_deadline_even_without_event_timer(self):
        clock = [0.0]
        stop = threading.Event()
        original = self.runtime.model.generate
        hint = self.runtime.external_hint
        def step(tokens, **kwargs):
            self.assertFalse(kwargs["stopping_criteria"][0](tokens, None))
            clock[0] = 301
            self.assertTrue(kwargs["stopping_criteria"][0](tokens, None))
            return original(tokens, **kwargs)
        with patch("luban.growth.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(self.runtime.model, "generate", side_effect=step):
            with self.assertRaises(TimeoutError):
                run_growth(self.runtime, self.challenge, stop, deadline_monotonic=300)
        self.assertFalse(stop.is_set())
        self.assertEqual(len(self.runtime.model.calls), 1)
        self.assertEqual(self.runtime.state, "ready")
        self.assertIs(self.runtime.external_hint, hint)

    def test_complete_answers_expiring_during_cold_release_are_not_returned(self):
        clock = [0.0]
        class Pager:
            releases = 0
            def acquire(self): pass
            def release(self):
                self.releases += 1
                if self.releases == 24:
                    clock[0] = 301
        self.runtime.cold_backbone = Pager()
        with patch("luban.growth.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                run_growth(self.runtime, self.challenge, deadline_monotonic=300)
        self.assertEqual(len(self.runtime.model.calls), 24)
        self.assertEqual(self.runtime.state, "ready")

    def test_native_core_helper_does_not_load_hf_layers_or_experts(self):
        self.runtime.cold_backbone = SimpleNamespace(acquire=lambda: self.fail("HF weights loaded"), release=lambda: None)
        self.runtime._load_active_expert = lambda: self.fail("core route loaded an expert")
        self.runtime.generate_tokens = lambda tokens, maximum, **kwargs: torch.cat((tokens, torch.tensor([[5, 9]])), dim=1)
        report = run_growth(self.runtime, self.challenge)
        self.assertEqual(len(report["answers"]), 24)
        self.assertFalse(report["metadata"]["branch_enabled"])

    def test_gated_off_saved_expert_does_not_load_hf_core(self):
        self.save_branch(requires_gate=True)
        self.save_router("local_core")
        self.challenge["model_version"] = personal_model_version(self.runtime)
        original = self.runtime.branch.expert
        self.runtime.cold_backbone = SimpleNamespace(acquire=lambda: self.fail("gated core loaded HF"), release=lambda: None)
        self.runtime._load_active_expert = lambda: self.fail("gated expert loaded")
        def generate_tokens(tokens, maximum, **kwargs):
            self.assertIsNone(self.runtime.branch.expert)
            return torch.cat((tokens, torch.tensor([[5, 9]])), dim=1)
        self.runtime.generate_tokens = generate_tokens
        report = run_growth(self.runtime, self.challenge)
        self.assertFalse(report["metadata"]["branch_enabled"])
        self.assertIs(self.runtime.branch.expert, original)

    def test_engine_change_invalidates_personal_version(self):
        self.runtime.inference_identity = lambda: {"engine": "luban-native", "binary_sha256": "f" * 64}
        self.assertNotEqual(personal_model_version(self.runtime), self.challenge["model_version"])
        with self.assertRaises(ValueError):
            run_growth(self.runtime, self.challenge)
        self.assertEqual(self.runtime.model.calls, [])

    def test_original_diagnostic_is_bounded_to_five_with_honest_count(self):
        for n in range(1, 7):
            self.source(f"Compute {n} + {7-n}.", "7")
        result = original_retest(self.runtime, eligible_cases(self.runtime))
        self.assertEqual(len(result["cases"]), 5)
        self.assertEqual(len(self.runtime.model.calls), 5)
        self.assertEqual(result["metadata"]["available_original_cases"], 6)
        self.assertEqual(result["metadata"]["tested_original_cases"], 5)
        self.assertFalse(result["counts_toward_public_growth"])

    def test_original_and_new_questions_share_the_supplied_deadline(self):
        self.source("Compute 3 + 4.", "7")
        clock = [0.0]
        stop = threading.Event()
        with patch("luban.growth.time.monotonic", side_effect=lambda: clock[0]):
            run_growth(self.runtime, self.challenge, stop, deadline_monotonic=150)
            clock[0] = 151
            with self.assertRaises(TimeoutError):
                original_retest(self.runtime, eligible_cases(self.runtime), stop, deadline_monotonic=150)
        self.assertEqual(len(self.runtime.model.calls), 24)

    def test_original_and_new_questions_share_cancellation(self):
        self.source("Compute 3 + 4.", "7")
        stop = threading.Event()
        run_growth(self.runtime, self.challenge, stop)
        stop.set()
        with self.assertRaises(InterruptedError):
            original_retest(self.runtime, eligible_cases(self.runtime), stop)
        self.assertEqual(len(self.runtime.model.calls), 24)

    def test_new_math_challenge_can_cover_unlearned_same_domain_skills(self):
        self.source("Compute 3 + 4.", "7")
        self.assertEqual(eligible_skills(self.runtime), ["arithmetic_add"])
        self.challenge["skills"] = ["arithmetic_add", "arithmetic_subtract", "arithmetic_multiply"]
        for index, question in enumerate(self.challenge["questions"]):
            question["skill"] = self.challenge["skills"][index % 3]
        result = run_growth(self.runtime, self.challenge)
        self.assertEqual(len(result["answers"]), 24)


if __name__ == "__main__":
    unittest.main()
