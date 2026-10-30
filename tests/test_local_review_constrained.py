from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_local_review_constrained.py"
SPEC = importlib.util.spec_from_file_location("constrained_review_protocol", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def perfect_results():
    values = []
    for row in review.cases():
        method = ("missing_reference" if row["mode"] == "unverified" else
                  "context_mismatch" if row["mode"] == "mismatch" else "local_reference_advisory_constrained")
        values.append({"id": row["id"], "result": {"verdict": row["expected"], "method": method,
            "raw": json.dumps(row["expected"]), "verified_source": False, "gate_enabled": False,
            "usage": {"finish_reason": "stop", "local_complete": True, "forced_completion": True,
                      "natural_completion_verified": False, "reference_used": False, "local_action": "local_core"}}})
    return values


class ConstrainedReviewTests(unittest.TestCase):
    def test_fresh_literal_references_counts_and_default_never_imports_model(self):
        rows, old = review.cases(), review.previous.cases()
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({r["id"] for r in rows}), 28)
        self.assertEqual([sum(r["expected"] is v for r in rows) for v in (True, False, None)], [10, 14, 4])
        self.assertFalse({r["source_text"] for r in rows} & {r["source_text"] for r in old})
        self.assertFalse({review.previous.encoded(r["context"]) for r in rows}
                         & {review.previous.encoded(r["context"]) for r in old})
        for row in rows:
            reference = review.reference_for(row)
            self.assertFalse(reference["approved"] or reference["training_allowed"])
            self.assertIs(reference["provenance"]["human_verified"], False)
            self.assertNotEqual(row["answer"].strip(), reference["answer"].strip())
        for flags in ([], ["--print-plan"]):
            done = subprocess.run([sys.executable, "-S", str(SCRIPT), *flags], capture_output=True, text=True, check=True)
            plan = json.loads(done.stdout)
            self.assertFalse(plan["model_go"] or plan["gate_enabled"] or plan["training_performed"])
            self.assertEqual(plan["teacher_calls"], 0)
            self.assertEqual(plan, review.plan())
        done = subprocess.run([sys.executable, "-S", str(SCRIPT), "--execute", "--output", "unused"],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertIn("requires --protocol", done.stderr)

    def test_semantic_thresholds_and_forced_eos_do_not_grant_truth(self):
        self.assertTrue(review.summarize(perfect_results())["protocol_passed"])
        values = perfect_results()
        values[0]["result"].update(verdict=None, raw="null")
        self.assertTrue(review.summarize(values)["protocol_passed"])
        values[7]["result"].update(verdict=None, raw="null")
        self.assertFalse(review.summarize(values)["protocol_passed"])
        for index in (1, 10, 24, 26, 27):
            values = perfect_results()
            values[index]["result"].update(verdict=None, raw="null") if index < 24 else values[index]["result"].update(verdict=False, raw="false")
            self.assertFalse(review.summarize(values)["protocol_passed"])
        for index in range(10, 28):
            values = perfect_results()
            values[index]["result"].update(verdict=True, raw="true")
            result = review.summarize(values)
            self.assertEqual(result["false_positives"], 1)
            self.assertFalse(result["protocol_passed"] or result["gate_enabled"])
        for changes in ({"raw": '{"true"}'}, {"verdict": 1}, {"verified_source": True}, {"gate_enabled": True}):
            values = perfect_results()
            values[0]["result"].update(changes)
            self.assertFalse(review.summarize(values)["protocol_passed"])
        for key, value in (("forced_completion", False), ("local_complete", False), ("finish_reason", "length"),
                           ("reference_used", True), ("local_action", "neural_expert"), ("natural_completion_verified", True)):
            values = perfect_results()
            values[0]["result"]["usage"][key] = value
            self.assertFalse(review.summarize(values)["protocol_passed"])
        for bad in (perfect_results()[:-1], perfect_results()[::-1]):
            with self.assertRaises(ValueError):
                review.summarize(bad)

    def test_draft_and_changed_source_are_rejected_before_model_import_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.json"
            path.write_bytes(review.previous.encoded(review.plan()))
            args = SimpleNamespace(protocol=str(path), expected_protocol_sha=review.previous.file_sha(path))
            with self.assertRaisesRegex(ValueError, "separately sealed"):
                review.execute(args)
            frozen = review.plan()
            frozen["execution_sources"]["src/luban/runtime.py"] = "0" * 64
            path.write_bytes(review.previous.encoded({"sealed": True, "state": "sealed_NOT_GO", "model_go": False, "plan": frozen}))
            args.expected_protocol_sha = review.previous.file_sha(path)
            with self.assertRaisesRegex(ValueError, "sources/cases/prompt differ"):
                review.execute(args)
            args.expected_protocol_sha = "0" * 64
            with self.assertRaisesRegex(ValueError, "protocol SHA differs"):
                review.execute(args)

    def test_explicit_local_descriptor_seals_without_rewriting_history_or_starting_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            descriptor = root / "independent-source-binding.json"
            descriptor.write_bytes(review.previous.encoded(review.plan()))
            historical = review.REPOSITORY / "research/local_review_constrained_protocol.json"
            original = historical.read_bytes()
            args = SimpleNamespace(descriptor=str(descriptor), output=str(root / "sealed"))
            def create(args):
                output = Path(args.output); output.mkdir()
                return root, output
            with patch.object(review, "new_output", side_effect=create):
                self.assertEqual(review.seal(args), 0)
            sealed = json.loads((root / "sealed/protocol.json").read_text())
            self.assertFalse(sealed["model_go"])
            self.assertEqual(sealed["binding_descriptor"], str(descriptor.resolve()))
            self.assertEqual(sealed["binding_descriptor_sha256"], review.previous.file_sha(descriptor))
            self.assertEqual(historical.read_bytes(), original)
            self.assertFalse((root / "sealed/execution-started.json").exists())
            args.descriptor = str(historical)
            args.output = str(root / "historical-mismatch")
            with self.assertRaisesRegex(ValueError, "Static descriptor differs"):
                review.seal(args)
            self.assertFalse(Path(args.output).exists())
            changed = review.plan(); changed["generation"] = {**changed["generation"], "max_tokens": 17}
            descriptor.write_bytes(review.previous.encoded(changed))
            args.descriptor = str(descriptor)
            with self.assertRaisesRegex(ValueError, "Static descriptor differs"):
                review.seal(args)
            # Even a hypothetically changed script matching a changed descriptor
            # cannot silently turn this rebind into a different semantic study.
            with patch.object(review, "plan", return_value=changed):
                with self.assertRaisesRegex(ValueError, "cannot change the fixed study"):
                    review.seal(args)
            self.assertEqual(historical.read_bytes(), original)

    def test_descriptor_cli_rejects_remote_missing_and_nonseal_modes_without_model_import(self):
        for flags in (["--seal", "--output", "unused", "--descriptor", "https://example.invalid/protocol.json"],
                      ["--seal", "--output", "unused", "--descriptor", "/definitely-absent/descriptor.json"],
                      ["--print-plan", "--descriptor", str(review.REPOSITORY / "research/local_review_constrained_protocol.json")]):
            done = subprocess.run([sys.executable, "-S", str(SCRIPT), *flags], capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn("--descriptor", done.stderr)

    def test_native_hf_processor_masks_invalid_highest_token_and_requires_eos(self):
        labels, eos, prefix = [4, 5, 6], [7, 8], torch.tensor([[1, 2, 3]])
        fn = lambda batch, ids: review.next_allowed_tokens(ids, 3, labels, eos)
        processor = PrefixConstrainedLogitsProcessor(fn, num_beams=1)
        scores = torch.tensor([[0., 0., 0., 100., 2., 8., 1., 9., 3., 200.]])
        masked = processor(prefix, scores)
        self.assertEqual(int(masked.argmax()), 5)
        self.assertTrue(torch.isneginf(masked[0, [0, 1, 2, 3, 7, 8, 9]]).all())
        masked = processor(torch.tensor([[1, 2, 3, 5]]), scores)
        self.assertEqual(int(masked.argmax()), 7)
        for invalid in ([1, 2, 3, 9], [1, 2, 3, 5, 7]):
            with self.assertRaises(ValueError):
                fn(0, torch.tensor(invalid))
        # EOS comes from model generation_config, never tokenizer's different ID.
        self.assertEqual(review.eos_token_ids(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[8, 7, 8]))), [8, 7])
        for bad in (None, True, [], [True], [-1]):
            with self.assertRaises(ValueError):
                review.eos_token_ids(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=bad)))

    def tokenizer(self):
        mapping = {"true": 4, "false": 5, "null": 6}
        def decode(ids, skip_special_tokens=False):
            return "".join({4: "true", 5: "false", 6: "null", 7: "", 8: ""}.get(int(i), "bad") for i in ids)
        return SimpleNamespace(encode=lambda text, **kwargs: [mapping[text]], decode=decode, eos_token_id=8)

    def runtime(self, *, throw=False, cancelled=False):
        value = SimpleNamespace(lock=threading.RLock(), state="ready", training_session=None, research=True,
            branch=SimpleNamespace(expert=None), active_core=None, device="cpu", tokenizer=self.tokenizer(),
            _ready=Mock(), _tokens=Mock(return_value=torch.tensor([[1, 2, 3]])))
        def generate(tokens, **kwargs):
            self.last_kwargs = kwargs
            self.assertEqual(value.state, "generating")
            self.assertTrue(value.lock._is_owned())
            if throw:
                raise RuntimeError("CPU synthetic generate failure")
            if cancelled:
                return torch.tensor([[1, 2, 3, 5]])
            processor = PrefixConstrainedLogitsProcessor(kwargs["prefix_allowed_tokens_fn"], num_beams=1)
            scores = torch.tensor([[0., 0., 0., 100., 2., 8., 1., 9., 3., 200.]])
            first = int(processor(tokens, scores).argmax())
            out = torch.cat((tokens, torch.tensor([[first]])), dim=1)
            second = int(processor(out, scores).argmax())
            return torch.cat((out, torch.tensor([[second]])), dim=1)
        value.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=7), generate=Mock(side_effect=generate))
        return value

    def test_compare_native_kwargs_raw_eos_and_cleanup_even_on_error(self):
        from luban import coldpaging
        row = review.cases()[0]
        entered, exited = [], []
        @contextmanager
        def full(runtime):
            entered.append(runtime)
            try:
                yield
            finally:
                exited.append(runtime)
        for throw in (False, True):
            runtime = self.runtime(throw=throw)
            with patch.object(coldpaging, "full_backbone", full):
                if throw:
                    with self.assertRaisesRegex(RuntimeError, "synthetic generate failure"):
                        review.compare_constrained(runtime, row["context"], row["answer"], review.reference_for(row))
                else:
                    result = review.compare_constrained(runtime, row["context"], row["answer"], review.reference_for(row))
                    self.assertIs(result["verdict"], False)
                    self.assertEqual(result["raw_token_ids"], [5, 7])
                    self.assertEqual(result["eos_token_ids"], [7])
                    self.assertFalse(result["verified_source"] or result["gate_enabled"])
                    self.assertTrue(result["usage"]["forced_completion"])
                    self.assertFalse(result["usage"]["natural_completion_verified"] or result["usage"]["reference_used"])
                    self.assertEqual(self.last_kwargs["max_new_tokens"], 16)
                    self.assertFalse(self.last_kwargs["do_sample"])
                    runtime._tokens.assert_called_once_with(unittest.mock.ANY, "off", output_reserve=16)
            self.assertEqual(runtime.state, "ready")
            self.assertIsNone(runtime.branch.expert)
        self.assertEqual(entered, exited)
        runtime = self.runtime()
        runtime._tokens.side_effect = ValueError("CPU admission fixture rejects budget")
        with patch.object(coldpaging, "full_backbone", full):
            with self.assertRaisesRegex(ValueError, "rejects budget"):
                review.compare_constrained(runtime, row["context"], row["answer"], review.reference_for(row))
        self.assertEqual(runtime.state, "ready")
        runtime.model.generate.assert_not_called()
        self.assertEqual(entered, exited)

    def test_outer_failure_releases_mock_runtime_and_keeps_error_report(self):
        from luban import runtime as runtime_module
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "run"
            sealed_dir = root / "experiments/sealed"; sealed_dir.mkdir(parents=True)
            protocol = sealed_dir / "protocol.json"
            protocol.write_bytes(review.previous.encoded({"sealed": True, "state": "sealed_NOT_GO",
                "model_go": False, "plan": review.plan()}))
            args = SimpleNamespace(protocol=str(protocol), expected_protocol_sha=review.previous.file_sha(protocol),
                expected_identity="a" * 64, expected_core="original", model=str(root / "CPU-fixture-never-loaded"))
            fake = SimpleNamespace(load=Mock(side_effect=RuntimeError("CPU mocked load failure")), _release_model=Mock())
            with patch.object(review.previous, "model_identity", return_value="a" * 64), \
                 patch.object(review, "new_output", return_value=(root, output)), \
                 patch.object(runtime_module, "Runtime", return_value=fake):
                with self.assertRaisesRegex(RuntimeError, "mocked load failure"):
                    review.execute(args)
            fake._release_model.assert_called_once_with()
            report = json.loads((output / "result.json").read_text())
            self.assertEqual(report["error"], "CPU mocked load failure")
            self.assertFalse(report["complete"] or report["training_performed"] or report["gate_enabled"])
            self.assertEqual(report["teacher_calls"], 0)

    def test_same_sealed_protocol_consumed_once_even_with_another_output(self):
        from luban import runtime as runtime_module, settings
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sealed_dir = root / "experiments/sealed"; sealed_dir.mkdir(parents=True)
            protocol = sealed_dir / "protocol.json"
            protocol.write_bytes(review.previous.encoded({"sealed": True, "state": "sealed_NOT_GO",
                "model_go": False, "plan": review.plan()}))
            args = SimpleNamespace(protocol=str(protocol), expected_protocol_sha=review.previous.file_sha(protocol),
                expected_identity="a" * 64, expected_core="original", model=str(root / "never-loaded"),
                storage=str(root), output=str(root / "experiments/first"))
            fake = SimpleNamespace(load=Mock(side_effect=RuntimeError("mocked interruption after claim")), _release_model=Mock())
            with patch.object(review.previous, "model_identity", return_value="a" * 64), \
                 patch.object(settings, "require_storage", return_value=root), \
                 patch.object(runtime_module, "Runtime", return_value=fake) as constructor:
                with self.assertRaisesRegex(RuntimeError, "interruption after claim"):
                    review.execute(args)
                marker = sealed_dir / "execution-started.json"
                data = json.loads(marker.read_text())
                self.assertEqual(data["output"], str(Path(args.output).resolve()))
                self.assertEqual(data["protocol_sha256"], args.expected_protocol_sha)
                self.assertEqual(data["expected_identity"], args.expected_identity)
                self.assertEqual(data["expected_core"], "original")
                before = marker.read_bytes()
                args.output = str(root / "experiments/second")
                with self.assertRaises(FileExistsError):
                    review.execute(args)
                self.assertFalse(Path(args.output).exists())
                self.assertEqual(marker.read_bytes(), before)
                constructor.assert_called_once()
            self.assertFalse(list(sealed_dir.glob(".core-precheck-*")))

    def test_wrong_source_core_identity_or_existing_output_do_not_consume(self):
        from luban import settings
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sealed_dir = root / "experiments/sealed"; sealed_dir.mkdir(parents=True)
            protocol = sealed_dir / "protocol.json"
            good = {"sealed": True, "state": "sealed_NOT_GO", "model_go": False, "plan": review.plan()}
            args = SimpleNamespace(protocol=str(protocol), expected_identity="a" * 64, expected_core="original",
                model=str(root / "never-loaded"), storage=str(root), output=str(root / "experiments/new"))
            def write(value):
                protocol.write_bytes(review.previous.encoded(value))
                args.expected_protocol_sha = review.previous.file_sha(protocol)
            with patch.object(settings, "require_storage", return_value=root), \
                 patch.object(review.previous, "model_identity", return_value="a" * 64) as identity:
                changed = {**good, "plan": {**good["plan"], "judge_prompt": "changed"}}
                write(changed)
                with self.assertRaisesRegex(ValueError, "sources/cases/prompt differ"):
                    review.execute(args)
                write(good)
                identity.return_value = "b" * 64
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    review.execute(args)
                identity.return_value = "a" * 64
                store = root / "modules/luban0-v1"; store.mkdir(parents=True)
                (store / "active-core.json").write_text('{"version":"core-123456789abc"}')
                with self.assertRaisesRegex(ValueError, "Selected core differs"):
                    review.execute(args)
                self.assertFalse(Path(args.output).exists())
                (store / "active-core.json").unlink()
                Path(args.output).mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    review.execute(args)
            self.assertFalse((sealed_dir / "execution-started.json").exists())
            self.assertFalse(list(sealed_dir.glob(".core-precheck-*")))

    def test_busy_branch_and_missing_reference_never_generate(self):
        from luban import coldpaging
        row = review.cases()[0]
        for changes in ({"state": "generating"}, {"training_session": object()}, {"research": False}):
            runtime = self.runtime()
            for key, value in changes.items():
                setattr(runtime, key, value)
            with self.assertRaises(RuntimeError):
                review.compare_constrained(runtime, row["context"], row["answer"], review.reference_for(row))
            runtime.model.generate.assert_not_called()
        runtime = self.runtime()
        runtime.branch.expert = object()
        with self.assertRaises(RuntimeError):
            review.compare_constrained(runtime, row["context"], row["answer"], review.reference_for(row))
        runtime.model.generate.assert_not_called()
        for unknown in review.cases()[-4:-2]:
            runtime = self.runtime()
            result = review.compare_constrained(runtime, unknown["context"], unknown["answer"], review.reference_for(unknown))
            self.assertIsNone(result["verdict"])
            runtime.model.generate.assert_not_called()
        runtime = self.runtime(cancelled=True)
        stop = threading.Event(); stop.set()
        with patch.object(coldpaging, "full_backbone", lambda runtime: __import__("contextlib").nullcontext()):
            result = review.compare_constrained(runtime, row["context"], row["answer"], review.reference_for(row), stop=stop)
        self.assertEqual(result["usage"]["finish_reason"], "cancelled")
        self.assertFalse(result["usage"]["forced_completion"])
        self.assertIsNone(result["verdict"])
        self.assertEqual(runtime.state, "ready")

    def test_tokenizer_validation_cpu_local_serialization_no_model(self):
        toy = self.tokenizer()
        self.assertEqual(review.label_token_ids(toy), [4, 5, 6])
        for encode in (lambda text, **kw: [1, 2], lambda text, **kw: [1], lambda text, **kw: [True]):
            with self.assertRaises(ValueError):
                review.label_token_ids(SimpleNamespace(encode=encode, decode=lambda ids, **kw: "true"))
        # Tokenizer-only preflight is optional on other machines; no weight load.
        path = Path("/Volumes/移动数据库/lubanllm/models/luban0/tokenizer.json")
        if path.is_file():
            native = Tokenizer.from_file(str(path))
            tokenizer = SimpleNamespace(encode=lambda text, **kw: native.encode(text, **kw).ids,
                                        decode=lambda ids, **kw: native.decode(ids, **kw))
            ids = review.label_token_ids(tokenizer)
            self.assertEqual(len(ids), 3)
            self.assertEqual(len(set(ids)), 3)


if __name__ == "__main__":
    unittest.main()
