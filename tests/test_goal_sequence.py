"""CPU tiny-state tests for the real goal-sequence module; no model/worker load."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from luban import goal_sequence as g
from luban.evolution import _consolidate_loaded, _resume_core


def fixture():
    def sample(sample_id):
        return {"id": sample_id, "answer": "unchanged exact gold", "question": sample_id}
    records = {"groups": {"new_training": [sample("new" + str(i)) for i in range(16)],
                          "new_validation": [sample("valid" + str(i)) for i in range(4)],
                          "replay_training": [sample("oldtrain" + str(i)) for i in range(83)],
                          "replay_validation": [sample("oldvalid" + str(i)) for i in range(8)]},
               "protection": [sample("protection" + str(i)) for i in range(89)],
               "retention_regressions": [sample("accept" + str(i)) for i in range(8)]}
    original = torch.zeros((2, 4), dtype=torch.bfloat16)
    master = torch.nn.Parameter(original.float().clone())
    optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
    basis = torch.tensor([[1.], [0.], [0.], [0.]])
    for i in range(450):
        optimizer.zero_grad(set_to_none=True)
        master.grad = torch.full_like(master, .1 + (i % 7) * .01)
        optimizer.step()
        with torch.no_grad():
            master[:, 0] = 0
    native = {"candidate": master.detach().to(torch.bfloat16).clone(), "master": master.detach().clone(),
              "reference": original.float().clone(), "basis": basis.clone(),
              "exp_avg": optimizer.state[master]["exp_avg"].clone(),
              "exp_avg_sq": optimizer.state[master]["exp_avg_sq"].clone(),
              "adam_step": optimizer.state[master]["step"].clone()}
    old_id = "77cff05a26ee"
    replay_ids = ["oldtrain" + str(i) for i in range(13)]
    def observation(sample_id, passed):
        return {"id": sample_id, "passed": passed, "complete": passed, "branch_enabled": False,
                "answer": "actual preserved output", "raw_generated_token_ids": [123, 124]}
    results = [{"step": step, "new_generation": [observation(r["id"], False) for r in records["groups"]["new_training"]],
                "replay_training_generation": [observation(i, True) for i in replay_ids], "replay_loss": 1.,
                "accepted": False, "minimum_new_exposures": ((step + 1) // 2) // 16,
                "consecutive_passes": 0, "constraint_residual": {"master_frobenius": 0.}}
               for step in g.HISTORY]
    report = {"id": old_id, "version": "core-" + old_id, "identity": "identity",
              "previous_version": "core-b45d965728bb", "promoted": False, "steps": 450,
              "dataset_sha256": g.content_sha(records["groups"]), "parameter": "target.down_proj.weight",
              "partial_policy": None, "baseline_training_replay_loss": 1., "baseline_core_retention_generation": [],
              "training_engine": {"engine": "native-test", "binary_sha256": "a" * 64},
              "training_selection": {"checkpoints": g.HISTORY.copy(), "results": results, "selected_step": None,
                  "required_consecutive_passes": 2, "required_new_exposures": 2, "replay_generation_ids": replay_ids,
                  "minimum_replay_generation": 13, "policy_version": "original-product"},
              "replay_subspace": {"rank": 1}, "started": 123., "split_ids": {}, "field_loss_policy": None}
    old_binding = {"identity": "identity", "scope": "product", "core_version": "core-b45d965728bb",
                   "device": "mps", "dtype": "torch.bfloat16", "arguments": g.BASE_ARGUMENTS.copy(),
                   "active_files_sha256": {"active-core.json": "c" * 64}, "expert_id": None,
                   "records_sha256": g.content_sha(records), "implementation_sha256": {"evolution.py": "d" * 64}}
    old = {"format": 1, "id": old_id, "step": 450, "master": master.detach().clone(),
           "optimizer": copy.deepcopy(optimizer.state_dict()), "basis": basis.clone(),
           "original_sha256": g.tensor_sha(original), "dataset_sha256": report["dataset_sha256"],
           "parameter": report["parameter"], "new_exposures": {"new" + str(i): 14 + int(i == 0) for i in range(16)},
           "kl_total": 1., "kl_maximum": .01, "consecutive_passes": 0, "report": report}
    closed = {"binding": old_binding, "payload": old, "rng": torch.get_rng_state().clone(),
              "mps_rng": torch.tensor([1, 2, 3], dtype=torch.uint8)}
    closure = {"exit_code": 1, "error_type": "ValueError", "error": "隔离候选没有通过正式核心训练门禁",
               "released": True, "adopted": False, "production_written": False, "validation_reads": 0,
               "final_step": 450, "parent_journal_state": "rejected", "active_core": old_binding["core_version"]}
    protocol = {"protocol": g.PROTOCOL, "status": "FROZEN_CLOSED_PARENT", "parent_steps": 450,
                "max_steps": 1800, "segment_steps": 450, "max_segments": 4, "checkpoints": g.CHECKPOINTS.copy(),
                "parent_arguments": g.BASE_ARGUMENTS.copy(), "seed": 42, "batch_size": 1, "gradient_accumulation": 1,
                "parent_run_id": old_id, "new_run_id": "abcdef123456", "policy_version": "goal-fixed450-same-weight-v2",
                "required_consecutive_passes": 0, "required_same_weight_passes": 2, "required_new_exposures": 2,
                "reference_policy": "original-accepted-core", "basis_policy": "exact-parent-basis",
                "validation_policy": "first-complete450-same-W-double-pass-once-then-stop", "_protocol_sha256": "0" * 64,
                "starting_core": old_binding["core_version"], "identity": old_binding["identity"],
                "parent_binding_sha256": g.content_sha(old_binding), "parent_report_sha256": g.content_sha(report),
                "records_sha256": g.content_sha(records), "dataset_sha256": report["dataset_sha256"],
                "parameter": report["parameter"], "engine": "native-test", "binary_sha256": "a" * 64,
                "replay_generation_ids": replay_ids, "original_tensor_sha256": old["original_sha256"],
                "parent_final_tensor_sha256": g.tensor_sha(native["candidate"]),
                "basis_sha256": g.tensor_sha(basis), "reference_sha256": g.tensor_sha(native["reference"]),
                "parent_implementation_sha256": old_binding["implementation_sha256"],
                "new_implementation_sha256": {"evolution.py": "e" * 64, "goal_sequence.py": "f" * 64},
                "closure": closure, "files": {name: {"sha256": "a" * 64} for name in
                    ("parent_checkpoint", "parent_journal", "native_output_manifest")}}
    new_binding = copy.deepcopy(old_binding)
    new_binding.update(arguments=g.arguments(protocol), implementation_sha256=protocol["new_implementation_sha256"])
    return protocol, closed, native, new_binding, records, closure


class GoalSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture()

    def setUp(self):
        self.values = copy.deepcopy(self.base)
        self.shape = patch.object(g, "SHAPE", (2, 4))  # Only tests change the real 2048x6144 constant.
        self.shape.start()
        self.addCleanup(self.shape.stop)

    def derive(self):
        return g.derive_payload(*self.values)

    def reject(self, change):
        change(*self.values)
        with self.assertRaises((ValueError, KeyError)):
            self.derive()

    def test_valid_distinct_payload_preserves_state_history_and_original(self):
        p, closed, native, binding, records, closure = self.values
        before = copy.deepcopy(closed)
        derived = self.derive()
        self.assertEqual(derived["id"], p["new_run_id"])
        self.assertEqual(derived["step"], 450)
        self.assertEqual(derived["report"]["steps"], 1800)
        self.assertEqual(derived["report"]["training_selection"]["checkpoints"], g.CHECKPOINTS)
        self.assertEqual(derived["report"]["training_selection"]["results"], before["payload"]["report"]["training_selection"]["results"])
        self.assertEqual(closed["payload"]["report"], before["payload"]["report"])
        self.assertTrue(torch.equal(derived["master"], before["payload"]["master"]))
        self.assertIsNot(derived["master"], closed["payload"]["master"])
        derived["master"].fill_(99)
        self.assertTrue(torch.equal(closed["payload"]["master"], before["payload"]["master"]))

    def test_complete_adam_resume_next_update_matches_uninterrupted_bitwise(self):
        derived = self.derive()
        masters, optimizers = [], []
        for state in (self.values[1]["payload"], derived):
            master = torch.nn.Parameter(state["master"].clone())
            optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
            optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
            master.grad = torch.full_like(master, .123)
            optimizer.step()
            masters.append(master)
            optimizers.append(optimizer)
        self.assertTrue(torch.equal(*masters))
        self.assertEqual(float(optimizers[1].state[masters[1]]["step"]), 451.)
        self.assertTrue(torch.equal(optimizers[0].state[masters[0]]["exp_avg"], optimizers[1].state[masters[1]]["exp_avg"]))

    def test_derived_payload_passes_actual_unchanged_resume_core(self):
        from types import SimpleNamespace
        derived = self.derive()
        parameter = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.bfloat16))
        runtime = SimpleNamespace(branch=SimpleNamespace(expert=None))
        def generation(runtime, rows):
            return [{"id": r["id"], "passed": r["id"].startswith("oldtrain"), "complete": True,
                     "branch_enabled": False, "reference_used": False} for r in rows]
        derived["report"]["goal_sequence"].update(g.entry_observations(runtime, self.values[0], derived,
            parameter, self.values[2]["reference"].to(torch.bfloat16), self.values[4]["groups"], generation, lambda *_: 1.))
        derived["report"]["goal_sequence"]["stage_observations"] = []
        g.classify_round(derived["report"]["goal_sequence"]["entry450"], 1.06)
        master = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.float32))
        optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
        report = _resume_core(SimpleNamespace(id=derived["id"]), derived, copy.deepcopy(derived["report"]),
            parameter, self.values[2]["reference"].to(torch.bfloat16), self.values[4]["groups"],
            1800, master, optimizer)
        self.assertEqual(report["goal_sequence"]["parent_steps"], 450)
        self.assertTrue(torch.equal(master.detach(), self.values[2]["master"]))
        self.assertTrue(torch.equal(parameter.detach(), self.values[2]["candidate"]))
        self.assertEqual(float(optimizer.state[master]["step"]), 450.)
        self.assertTrue(torch.equal(optimizer.state[master]["exp_avg_sq"], self.values[2]["exp_avg_sq"]))

    def test_budget_policy_rejects_generalized_or_bool_or_changed_schedule(self):
        for key, value in (("max_steps", 2250), ("max_segments", 5), ("parent_steps", True),
                           ("batch_size", True), ("seed", 43), ("checkpoints", g.CHECKPOINTS + [1832]),
                           ("reference_policy", "parent-candidate-reference")):
            with self.subTest(key=key):
                p = copy.deepcopy(self.values[0]); p[key] = value
                with self.assertRaises(ValueError): g.validate_protocol(p)

    def test_reject_448_and_wrong_old_journal_state(self):
        self.reject(lambda p,c,n,b,r,x: c["payload"].update(step=448))
        self.setUp()
        self.reject(lambda p,c,n,b,r,x: x.update(parent_journal_state="running"))

    def test_reject_unclosed_or_unexpected_process_error(self):
        self.reject(lambda p,c,n,b,r,x: x.update(released=False))
        self.setUp()
        self.reject(lambda p,c,n,b,r,x: x.update(error="native worker crashed"))

    def test_reject_heldout_read(self):
        self.reject(lambda p,c,n,b,r,x: x.update(validation_reads=1))

    def test_reject_float32_master_roundtrip_loss(self):
        self.reject(lambda p,c,n,b,r,x: c["payload"].update(master=c["payload"]["master"].to(torch.bfloat16).float()))

    def test_reject_missing_adam_or_reset_moments(self):
        self.reject(lambda p,c,n,b,r,x: n.pop("exp_avg"))
        self.setUp()
        self.reject(lambda p,c,n,b,r,x: n["exp_avg"].zero_())

    def test_reject_different_reference_or_basis(self):
        self.reject(lambda p,c,n,b,r,x: n["reference"].fill_(1))
        self.setUp()
        self.reject(lambda p,c,n,b,r,x: n["basis"].fill_(.5))

    def test_reject_nonfinite_or_negative_second_moment(self):
        self.reject(lambda p,c,n,b,r,x: n["master"].fill_(float("nan")))
        self.setUp()
        self.reject(lambda p,c,n,b,r,x: n["exp_avg_sq"].fill_(-1))

    def test_reject_changed_identity_device_core_or_sources(self):
        for key, value in (("identity", "other"), ("device", "cpu"), ("core_version", "core-000000000000"),
                           ("implementation_sha256", {"evolution.py": "9" * 64})):
            with self.subTest(key=key):
                values = copy.deepcopy(self.base); values[3][key] = value
                with self.assertRaises(ValueError): g.derive_payload(*values)

    def test_reject_changed_gold_or_removed_old_protection(self):
        self.reject(lambda p,c,n,b,r,x: r["groups"]["new_training"][0].update(answer="wrong gold"))
        self.setUp()
        self.reject(lambda p,c,n,b,r,x: r["protection"].pop())

    def test_reject_parent_report_hash_tampering(self):
        self.reject(lambda p,c,n,b,r,x: c["payload"]["report"]["training_selection"]["results"].pop())

    def test_reject_schedule_exposure_or_rng_reset(self):
        self.reject(lambda p,c,n,b,r,x: c["payload"]["new_exposures"].update(new0=14))
        self.setUp()
        self.reject(lambda p,c,n,b,r,x: c.update(mps_rng=None))

    def test_progress_counter_and_binding_stable(self):
        p = self.values[0]
        before = g.arguments(p)
        self.assertEqual(g.report_metadata(p, 900)["selected_segment"], 2)
        self.assertEqual(g.report_metadata(p, 868)["new_steps_executed"], 418)
        self.assertEqual(g.report_metadata(p, 1800)["cumulative_steps"], 1800)
        self.assertEqual(g.arguments(p), before)
        with self.assertRaises(ValueError): g.report_metadata(p, 1801)

    def test_real450_observation_and_later_reference_names_restore_starting_w(self):
        from types import SimpleNamespace
        p, closed, native, binding, records, closure = self.values
        expert = object()
        runtime = SimpleNamespace(branch=SimpleNamespace(expert=expert))
        parameter = torch.nn.Parameter(native["reference"].to(torch.bfloat16).clone())
        before = parameter.detach().clone()
        accessed = []
        def generation(runtime, rows):
            self.assertIsNone(runtime.branch.expert)
            accessed.extend(row["id"] for row in rows)
            moved = float(parameter[0, 1]) != 0
            return [{"id": row["id"], "passed": row["id"].startswith("oldtrain") or moved and row["id"] in {"new0", "new1", "new2"},
                     "complete": True, "branch_enabled": False, "reference_used": False} for row in rows]
        proof = g.entry_observations(runtime, p, closed["payload"], parameter,
            native["reference"].to(torch.bfloat16), records["groups"], generation, lambda runtime, rows: 1.)
        self.assertEqual(proof["reference_baseline_after_parent_closed"]["confirmed_count"], 0)
        self.assertEqual(proof["entry450"]["confirmed_count"], 3)
        self.assertEqual(proof["entry450"]["tensor_sha256"], g.tensor_sha(native["candidate"]))
        self.assertFalse(any(sample_id.startswith(("valid", "oldvalid", "protection", "accept")) for sample_id in accessed))
        self.assertEqual(len(accessed), 2 * 2 * (16 + 13))
        self.assertTrue(torch.equal(parameter, before))
        self.assertIs(runtime.branch.expert, expert)

    def test_boundary_intersection_counts_and_forgetting_are_separate(self):
        from types import SimpleNamespace
        p, closed, native, binding, records, closure = self.values
        runtime = SimpleNamespace(branch=SimpleNamespace(expert=None))
        parameter = torch.nn.Parameter(native["candidate"].clone())
        calls = 0
        def generation(runtime, rows):
            nonlocal calls
            calls += 1
            return [{"id": row["id"], "passed": row["id"].startswith("oldtrain") or row["id"] == "new0"
                     or row["id"] == "new1" and calls == 3, "complete": True,
                     "branch_enabled": False, "reference_used": False} for row in rows]
        proof = g.observe_training_state(runtime, records["groups"], p["replay_generation_ids"], parameter, 900,
            generation, lambda runtime, rows: 1., reference_confirmed=["new0"], prior_confirmed=["new0", "new5"])
        self.assertEqual(proof["single_observation_correct"], [1, 2])
        self.assertEqual(proof["confirmed_ids"], ["new0"])
        self.assertEqual(proof["forgotten_since_prior"], ["new5"])
        self.assertEqual(proof["newly_confirmed_from_reference"], [])
        self.assertEqual(proof["newly_confirmed_since_prior"], [])

    def test_observation_error_or_w_mutation_always_restore_entry_parameter_and_expert(self):
        from types import SimpleNamespace
        p, closed, native, binding, records, closure = self.values
        expert = object()
        runtime = SimpleNamespace(branch=SimpleNamespace(expert=expert))
        parameter = torch.nn.Parameter(native["reference"].to(torch.bfloat16).clone())
        before = parameter.detach().clone()
        def broken_generation(runtime, rows):
            with torch.no_grad(): parameter.fill_(99)
            raise RuntimeError("CPU injected generation failure")
        with self.assertRaises(RuntimeError):
            g.entry_observations(runtime, p, closed["payload"], parameter, before, records["groups"],
                broken_generation, lambda runtime, rows: 1.)
        self.assertTrue(torch.equal(parameter, before))
        self.assertIs(runtime.branch.expert, expert)

    def test_load_verified_rejects_symlink_bad_sha_or_unknown_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "protocol.json"
            path.write_text(json.dumps(self.values[0]))
            symlink = Path(temp) / "alias.json"; symlink.symlink_to(path)
            with self.assertRaises(ValueError): g.load_verified(symlink)
            with self.assertRaises(ValueError): g.load_verified(path)

    def test_real_file_closure_then_tamper_and_missing_source_are_rejected(self):
        p, closed, native, binding, records, closure = self.values
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            def write(name, value):
                path = root / name
                path.write_text(json.dumps(value, ensure_ascii=False))
                return {"path": str(path), "sha256": g.file_sha(path)}
            snapshots = {}
            for i in range(19):
                name = "src/luban/source" + str(i) + ".py"
                snapshots[name] = write("source" + str(i) + ".py", "cpu-test-original-source-" + str(i))
            impl = {Path(name).name: entry["sha256"] for name, entry in snapshots.items()}
            closed["binding"]["implementation_sha256"] = impl
            p.update(parent_implementation_sha256=impl,
                     parent_binding_sha256=g.content_sha(closed["binding"]),
                     parent_source_snapshot=snapshots,
                     parent_implementation_snapshot={Path(name).name: entry for name, entry in snapshots.items()},
                     parent_source_sha256={name: entry["sha256"] for name, entry in snapshots.items()})
            helper_sha = g.file_sha(Path(g.__file__))
            p["new_implementation_sha256"] = {**impl, "goal_sequence.py": helper_sha}
            p["new_source_sha256"] = {**p["parent_source_sha256"], "src/luban/goal_sequence.py": helper_sha}
            files = {}
            for name in g.FORMAL:
                files[name] = write(name + ".json", {"cpu-test-only": name})
            fake_formal = {name: entry["sha256"] for name, entry in files.items()}
            plan = {"protocol": "evolution-goal-current-core-native-v1", "arguments": g.BASE_ARGUMENTS,
                    "seed": 42, "formal_protocol_sha256": fake_formal["formal_protocol"],
                    "materials_sha256": fake_formal["materials"], "identity": p["identity"],
                    "starting_core": p["starting_core"], "source_sha256": p["parent_source_sha256"]}
            files["parent_plan"] = write("plan.json", plan)
            files["parent_binding"] = write("binding.json", closed["binding"])
            files["parent_report"] = write("report.json", closed["payload"]["report"])
            files["records"] = write("records.json", records)
            files["parent_checkpoint"] = write("checkpoint.pt", "CPU file hashing fixture, not a model")
            journal = {"id": p["parent_run_id"], "state": "rejected", "binding": closed["binding"],
                       "records": records, "checkpoint": {"sha256": files["parent_checkpoint"]["sha256"]}}
            files["parent_journal"] = write("journal.json", journal)
            baseline = [{"id": row["id"], "answer": "preserved observation"} for row in records["groups"]["replay_training"]]
            files["baseline_old_training"] = write("baseline.json", baseline)
            final_report = copy.deepcopy(closed["payload"]["report"]); final_report["completed_steps"] = 450
            final_report["elapsed_seconds"] = 100.
            p["parent_training_elapsed_seconds"] = 100.
            result = {"released": True, "passed": False, "adopted": False, "production_written": False,
                      "plan_sha256": files["parent_plan"]["sha256"], "training": final_report,
                      "baseline_old_training": baseline, "error_type": closure["error_type"], "error": closure["error"]}
            files["parent_result"] = write("result.json", result)
            for name in ("native_manifest", "native_output_manifest", "native_trace"):
                files[name] = write(name + ".json", {"fixture": name})
            tensor = write("native-cache.f32", "CPU cache file hashing fixture")
            p.update(files=files, native_files={tensor["path"]: tensor["sha256"]})
            descriptor = write("protocol.json", p)
            with patch.object(g, "FORMAL", fake_formal):  # Never relaxes real formal constants.
                checked = g.load_verified(descriptor["path"])
                self.assertEqual(checked["_protocol_sha256"], descriptor["sha256"])
                Path(tensor["path"]).write_text("corrupted cache")
                with self.assertRaisesRegex(ValueError, "native tensor/cache SHA"):
                    g.load_verified(descriptor["path"])
                Path(tensor["path"]).write_text(json.dumps("CPU cache file hashing fixture"))
                p["parent_source_snapshot"].pop(next(iter(p["parent_source_snapshot"])))
                write("protocol.json", p)
                with self.assertRaisesRegex(ValueError, "every original plan source"):
                    g.load_verified(descriptor["path"])

    def test_json_duplicate_key_nonfinite_overflow_and_oversize_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "bad.json"
            for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
                path.write_text(raw)
                with self.assertRaises(ValueError): g.read_json(path)
            with path.open("wb") as stream: stream.truncate(32 * 1024**2 + 1)
            with self.assertRaisesRegex(ValueError, "bounded JSON"): g.read_json(path)

    def round_fixture(self):
        from types import SimpleNamespace
        p, closed, native, binding, records, closure = self.values
        saved = self.derive()
        runtime = SimpleNamespace(branch=SimpleNamespace(expert=None), job={})
        parameter = torch.nn.Parameter(native["candidate"].clone())
        def generation(runtime, rows):
            return [{"id": r["id"], "passed": r["id"].startswith("oldtrain"), "complete": True,
                     "branch_enabled": False, "reference_used": False} for r in rows]
        saved["report"]["goal_sequence"].update(g.entry_observations(runtime, p, saved, parameter,
            native["reference"].to(torch.bfloat16), records["groups"], generation, lambda *_: 1.))
        saved["report"]["goal_sequence"]["stage_observations"] = []
        g.classify_round(saved["report"]["goal_sequence"]["entry450"], 1.06)
        return runtime, parameter, saved

    def run_boundary(self, runtime, parameter, saved, step, mastered=(), *, old=True, loss=1.):
        def generation(runtime, rows):
            return [{"id": r["id"], "passed": old if r["id"].startswith("oldtrain") else r["id"] in mastered,
                     "complete": True, "branch_enabled": False, "reference_used": False} for r in rows]
        return g.observe_boundary(runtime, self.values[0], saved["report"], self.values[4]["groups"],
            parameter, step, generation, lambda *_: loss)

    def test_zero_and_partial_rounds_continue_then_full_boundary_selects(self):
        runtime, parameter, saved = self.round_fixture()
        proof = self.run_boundary(runtime, parameter, saved, 900)
        self.assertEqual((proof["round_result"], proof["confirmed_count"], proof["pending_count"]), ("pending", 0, 16))
        self.assertIsNone(saved["report"]["training_selection"]["selected_step"])
        self.assertFalse(g.stopped(saved["report"]))
        proof = self.run_boundary(runtime, parameter, saved, 1350, ["new0", "new1"])
        self.assertEqual((proof["round_result"], proof["pending_count"]), ("partial", 14))
        self.assertIsNone(saved["report"]["training_selection"]["selected_step"])
        self.assertFalse(g.stopped(saved["report"]))
        proof = self.run_boundary(runtime, parameter, saved, 1800, ["new"+str(i) for i in range(16)])
        self.assertEqual((proof["round_result"], proof["pending_count"]), ("satisfied", 0))
        self.assertEqual(saved["report"]["training_selection"]["selected_step"], 1800)
        self.assertEqual([p["actual_updates"] for p in saved["report"]["goal_sequence"]["stage_observations"]], [450]*3)
        self.assertEqual(saved["consecutive_passes"], 0)

    def test_first_full_round900_selects_without_temporal_double_check(self):
        runtime, parameter, saved = self.round_fixture()
        proof = self.run_boundary(runtime, parameter, saved, 900, ["new"+str(i) for i in range(16)])
        self.assertEqual(saved["report"]["training_selection"]["selected_step"], 900)
        self.assertEqual(saved["report"]["training_selection"]["results"][-1]["same_weight_passes"], 2)
        self.assertEqual(len(proof["observations"]), 2)
        with self.assertRaisesRegex(ValueError, "after protection failure or satisfaction"):
            self.run_boundary(runtime, parameter, saved, 1350)

    def test_forgetting_previous_mastery_stops_even_if_old13_passes(self):
        runtime, parameter, saved = self.round_fixture()
        self.run_boundary(runtime, parameter, saved, 900, ["new0"])
        proof = self.run_boundary(runtime, parameter, saved, 1350, ["new1"])
        self.assertEqual(proof["forgotten_since_prior"], ["new0"])
        self.assertEqual(proof["round_result"], "protection_failed")
        self.assertTrue(g.stopped(saved["report"]))
        self.assertIsNone(saved["report"]["training_selection"]["selected_step"])
        with self.assertRaises(ValueError): self.run_boundary(runtime, parameter, saved, 1800)

    def test_old13_or_loss_failure_has_priority_over_full_mastery(self):
        for old, loss in ((False, 1.), (True, 1.061)):
            with self.subTest(old=old, loss=loss):
                runtime, parameter, saved = self.round_fixture()
                proof = self.run_boundary(runtime, parameter, saved, 900, ["new"+str(i) for i in range(16)], old=old, loss=loss)
                self.assertEqual(proof["confirmed_count"], 16)
                self.assertEqual(proof["round_result"], "protection_failed")
                self.assertTrue(g.stopped(saved["report"]))
                self.assertIsNone(saved["report"]["training_selection"]["selected_step"])

    def test_early_checks_and_missing450_rounds_are_forbidden(self):
        runtime, parameter, saved = self.round_fixture()
        for step in (868, 1318, 1768, 1350):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.run_boundary(runtime, parameter, saved, step)
        self.assertEqual(g.CHECKPOINTS, g.HISTORY+[900,1350,1800])

    def test_actual_resume900_validates_same_weight_proof_and_ledger(self):
        from types import SimpleNamespace
        runtime, parameter, saved = self.round_fixture()
        self.run_boundary(runtime, parameter, saved, 900, ["new0"])
        saved["step"] = 900
        saved["new_exposures"] = {"new"+str(i): 28+int(i<2) for i in range(16)}
        declared = copy.deepcopy(saved["report"])
        master = torch.nn.Parameter(torch.zeros((2,4)))
        optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
        _resume_core(SimpleNamespace(id=saved["id"]), saved, declared, parameter,
            self.values[2]["reference"].to(torch.bfloat16), self.values[4]["groups"],1800,master,optimizer)
        for change in (lambda x: x["report"]["goal_sequence"]["stage_observations"][0].update(pending_count=0),
                       lambda x: x["report"]["training_selection"].update(selected_step=868),
                       lambda x: x.update(consecutive_passes=2),
                       lambda x: x["report"]["training_selection"]["results"][0].update(replay_loss=.5),
                       lambda x: x["report"]["goal_sequence"]["stage_observations"][0].update(tensor_sha256="0"*64)):
            altered=copy.deepcopy(saved); change(altered)
            with self.assertRaises(ValueError):
                _resume_core(SimpleNamespace(id=saved["id"]),altered,declared,parameter,
                    self.values[2]["reference"].to(torch.bfloat16),self.values[4]["groups"],1800,master,optimizer)

    def test_ordinary512_limit_and_unverified_goal_fails(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError, "64–512"):
            _consolidate_loaded(SimpleNamespace(), 1800, preserve_replay_subspace=True)
        with self.assertRaises((ValueError, TypeError)):
            _consolidate_loaded(SimpleNamespace(), 1800, preserve_replay_subspace=True,
                goal_sequence={"max_steps": 1800})


if __name__ == "__main__":
    unittest.main(verbosity=2)
