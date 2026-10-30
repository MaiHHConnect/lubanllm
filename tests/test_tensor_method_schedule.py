"""Fixed event coverage/indices/actualAdam/checkpoint counts; stdlib only."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_tensor_method_schedule import build_schedule


class TensorMethodScheduleTests(unittest.TestCase):
    def setUp(self):
        self.new = [f"new-{index}" for index in range(57)]
        self.replay = [f"replay-{index}" for index in range(57)]

    def test_all_nine_exact_checks_and_expected_adam_updates(self):
        counts = {150: {1:150, 2:78, 4:42}, 352: {1:352, 2:184, 4:100}, 704: {1:704, 2:366, 4:198}}
        for events, expected in counts.items():
            for size, updates in expected.items():
                with self.subTest(events=events, size=size):
                    report = build_schedule(self.new, self.replay, events, size)
                    self.assertEqual(report["optimizer_updates"], updates)
                    self.assertEqual(report["exposure_totals"], {"new_ce":events//2, "replay_ce":events//2, "replay_kl":events})
                    self.assertEqual(report["checks_ce_events"][-1], events)
                    cursor = 0
                    for segment in report["segments"]:
                        end = segment["check_ce_event"]
                        flat = [value for batch in segment["batches"] for value in batch["ce_event_ids"]]
                        self.assertEqual(sorted(flat), list(range(cursor + 1, end + 1)))
                        self.assertTrue(all(cursor < value <= end for value in flat))
                        self.assertTrue(all(1 <= batch["actual_batch_size"] <= size for batch in segment["batches"]))
                        self.assertEqual(segment["optimizer_updates_at_check"], segment["batches"][-1]["optimizer_update"])
                        cursor = end

    def test_role_order_cycle_wrap_and_odd25_tails_never_reset_indices(self):
        report = build_schedule(self.new, self.replay, 704, 4)
        physical = [batch for segment in report["segments"] for batch in segment["batches"]]
        for role, ids in (("new", self.new), ("replay", self.replay)):
            actual = [value for batch in physical if batch["ce_role"] == role for value in batch["ce_ids"]]
            self.assertEqual(actual, (ids * 7)[:352])
        by_event = {event: (batch["ce_role"], ce_id, kl_id, pair, index)
                    for batch in physical for event,ce_id,kl_id,pair,index in zip(batch["ce_event_ids"],batch["ce_ids"],batch["replay_kl_ids"],batch["pair_indices"],batch["sample_indices"])}
        self.assertEqual(by_event[51], ("new", "new-25", "replay-25", 25, 25))
        self.assertEqual(by_event[101], ("new", "new-50", "replay-50", 50, 50))
        self.assertEqual(by_event[114], ("replay", "replay-56", "replay-56", 56, 56))
        self.assertEqual(by_event[115], ("new", "new-0", "replay-0", 57, 0))
        self.assertEqual(by_event[151], ("new", "new-18", "replay-18", 75, 18))
        self.assertEqual(by_event[701], ("new", "new-8", "replay-8", 350, 8))
        self.assertEqual(by_event[704], ("replay", "replay-9", "replay-9", 351, 9))
        for segment in report["segments"][:-1]:
            self.assertEqual([batch["actual_batch_size"] for batch in segment["batches"][-2:]], [1,1])
        self.assertEqual([batch["actual_batch_size"] for batch in report["segments"][-1]["batches"]], [2,2])

    def test_equal_per_question_three_ledger_exposure_independent_of_batch(self):
        for events in (150,352,704):
            reports = [build_schedule(self.new,self.replay,events,size) for size in (1,2,4)]
            self.assertEqual(reports[0]["per_id_exposures"], reports[1]["per_id_exposures"])
            self.assertEqual(reports[0]["per_id_exposures"], reports[2]["per_id_exposures"])
            for report in reports:
                actual = {"new_ce":Counter(),"replay_ce":Counter(),"replay_kl":Counter()}
                for segment in report["segments"]:
                    for batch in segment["batches"]:
                        actual[batch["ce_role"]+"_ce"].update(batch["ce_ids"])
                        actual["replay_kl"].update(batch["replay_kl_ids"])
                self.assertEqual({key:dict(value) for key,value in actual.items()}, report["per_id_exposures"])
                for key in self.replay:
                    self.assertEqual(actual["replay_kl"][key], 2*actual["replay_ce"][key])

    def test_invalid_data_and_defaults_are_stdlib_only(self):
        for events,size in ((True,1),(150,True),(150,3),(352.,2),(151,4)):
            with self.subTest(events=events,size=size),self.assertRaises(ValueError):build_schedule(self.new,self.replay,events,size)
        for ids in (self.new[:-1],self.new[:-1]+[self.new[0]],self.new[:-1]+[""],self.new[:-1]+[None],"new"):
            with self.assertRaises(ValueError):build_schedule(ids,self.replay,150,2)
        with self.assertRaises(ValueError):build_schedule(self.new,self.new,150,2)
        script=ROOT/'scripts/prepare_tensor_method_schedule.py'
        result=subprocess.run([sys.executable,'-S',str(script)],check=True,capture_output=True,text=True)
        default=json.loads(result.stdout);self.assertFalse(default['torch_imported']);self.assertFalse(default['model_started'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ids.json';path.write_text(json.dumps({'new_ids':self.new,'replay_ids':self.replay}))
            actual=subprocess.run([sys.executable,'-S',str(script),'--ids-json',str(path),'--ce-events','150','--batch-size','2'],check=True,capture_output=True,text=True)
            report=json.loads(actual.stdout);self.assertEqual(report['optimizer_updates'],78);self.assertFalse(report['torch_imported'])
            self.assertFalse(report['model_started'])


if __name__ == '__main__':
    unittest.main()
