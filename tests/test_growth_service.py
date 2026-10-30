"""Prospective growth sequencing and holdout isolation, without production models."""
import copy
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from luban.growth_service import GrowthService, validated_growth
from luban.growth import POLICY_HASH
from luban.leaderboard import Leaderboard
from luban.memory import Memory

A, B, BASE = 'a'*64, 'b'*64, 'c'*64
NOW = datetime.now(timezone.utc)
RECEIPT = dict(receipt_id='d'*64, challenge_id='e'*64, suite_id='growth-v1', domain='mathematics',
               base_identity=BASE, before_model_version=A, model_version=B, policy_hash=POLICY_HASH,
               suite_sha256='1'*64, question_count=24, baseline_correct=12, current_correct=18,
               wrong_to_correct=9, correct_to_wrong=3, baseline_score=50, current_score=75,
               net_delta=25, gap_recovery=75, retention=75, eligible=True,
               baseline_at=NOW.isoformat(), completed_at=NOW.isoformat(), retention_check=None)


class GrowthServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.memory = Memory(self.root/'storage')
        self.runtime = SimpleNamespace(memory=self.memory, state='ready', model=object(), identity=BASE,
                                       training_session=None, _ready=lambda: None)
        self.version, self.calls = A, []
        self.board = Leaderboard(self.root/'home', storage_root=self.memory.root,
                                 transport=httpx.MockTransport(self.remote))
        self.board.save_profile({'username':'Growth tester'})
        self.service = GrowthService(self.runtime, self.board, threading.Lock(), lambda:False)
        self.board.evaluation_provider = self.service.profile
        self.addCleanup(self.service.close)
        patches = [patch('luban.growth_service.personal_model_version', side_effect=lambda r:self.version),
                   patch('luban.growth_service.model_version', return_value=BASE),
                   patch('luban.growth_service.eligible_skills', return_value=['arithmetic_add']),
                   patch('luban.growth_service.eligible_cases', return_value=[]),
                   patch('luban.growth_service.original_retest', return_value={'private':True,'counts_toward_public_growth':False}),
                   patch('luban.growth_service.run_growth', side_effect=self.generate)]
        for p in patches:p.start();self.addCleanup(p.stop)

    def remote(self, request):
        self.calls.append((request.url.path,json.loads(request.content)))
        if request.url.path.endswith('/challenge'):
            return httpx.Response(200,json=dict(challenge_id='e'*64,suite_id='growth-v1',domain='mathematics',
                model_version=self.version,base_identity=BASE,policy_hash=POLICY_HASH,suite_sha256='1'*64,
                questions=[dict(id=str(i),prompt=f'Reserved fresh task {i}') for i in range(24)]))
        if request.url.path.endswith('/before'):
            return httpx.Response(200,json=dict(challenge_id='e'*64,before_model_version=self.version,
                question_count=24,baseline_correct=12,baseline_score=50,
                expires_at=(NOW+timedelta(days=14)).isoformat()))
        if request.url.path.endswith('/after'):
            return httpx.Response(200,json=RECEIPT)
        if request.url.path.endswith('/retention'):
            return httpx.Response(200,json=dict(receipt_id=RECEIPT['receipt_id'],model_version=B,
                checked_at=datetime.now(timezone.utc).isoformat(),correct=20,retained_correct=17,retention=94.44))
        if request.url.path.endswith('/me'):
            body=json.loads(request.content)
            public={k:v for k,v in RECEIPT.items() if k not in
                    ('receipt_id','challenge_id','suite_id','skills','base_identity','before_model_version',
                     'model_version','policy_hash','suite_sha256','baseline_at')}
            return httpx.Response(200,json={'protocol_version':2,'entry':dict(username=body['username'],country_code=None,
                evolution_count=0,rank=1,growth=public,growth_profile=[{'domain':d,'growth':public if d=='mathematics' else None}
                for d in ('mathematics','programming','data')])})
        raise AssertionError(request.url.path)

    def generate(self, runtime, challenge, stop, progress, **kwargs):
        self.assertIsNone(runtime.training_session)
        return {'answers':[dict(id=str(i),text='1',finish_reason='stop') for i in range(24)],
                'metadata':{'items':[dict(prompt_tokens=2,completion_tokens=1)]}}

    def run_job(self):
        self.service.start();self.service.worker.join(3)
        self.assertFalse(self.service.worker.is_alive())
        self.assertNotEqual(self.service.state['state'],'error',self.service.state)

    def test_baseline_before_learning_then_changed_version_after_and_server_publication(self):
        self.run_job()
        self.assertEqual(self.service.state['state'],'baseline')
        self.assertFalse(any(p.endswith('/me') for p,_ in self.calls))
        self.assertEqual(self.service.profile()['growth'],None)
        self.version=B;self.run_job()
        self.assertEqual(self.service.state['state'],'complete')
        self.assertEqual(self.service.profile()['growth']['net_delta'],25)
        body=next(b for p,b in self.calls if p.endswith('/me'))
        self.assertEqual(body['growth_model_version'],B)
        self.assertEqual(body['model_version'],BASE)
        self.assertNotIn('score',body)
        self.assertEqual(len(self.memory.samples()),0)

    def test_same_version_baseline_does_not_rerun_or_publish(self):
        self.run_job(); before=len(self.calls)
        self.run_job()
        self.assertEqual(len(self.calls),before)
        self.assertEqual(self.memory.usage()['local']['total_tokens'],3)

    def test_completed_retry_does_not_regrade_or_repeat_tokens(self):
        self.run_job();self.version=B;self.run_job()
        previous=self.memory.usage()['local']['total_tokens']
        self.run_job()
        self.assertEqual(self.memory.usage()['local']['total_tokens'],previous)
        self.assertEqual(sum(p.endswith('/after') for p,_ in self.calls),1)

    def test_current_model_change_invalidates_public_profile(self):
        self.run_job();self.version=B;self.run_job();self.version='2'*64
        self.assertIsNone(self.service.profile()['growth'])
        self.assertTrue(all(v['growth'] is None for v in self.service.profile()['growth_profile']))

    def test_no_supported_source_does_not_send_requests_or_fake_zero_score(self):
        with patch('luban.growth_service.eligible_skills',return_value=[]):
            state=self.service.start()
        self.assertEqual(state['state'],'waiting_materials')
        self.assertEqual(self.calls,[])
        self.assertFalse(self.service.path.exists())

    def test_pretraining_queued_session_is_temporarily_detached_and_restored(self):
        self.run_job();data=self.service._read();data['active']=None;self.service._save(data)
        session=SimpleNamespace(body={'checkpoint':None},pause_requested=False)
        self.runtime.training_session=session
        self.service.before_training(session)
        self.assertIs(self.runtime.training_session,session)
        self.assertEqual(self.service.state['state'],'baseline')

    def test_resumed_training_cannot_backfill_a_baseline(self):
        data=self.service._read();data['enabled']=True;self.service._save(data)
        session=SimpleNamespace(body={'checkpoint':{'name':'existing.pt'}},pause_requested=False)
        self.service.before_training(session)
        self.assertEqual(self.calls,[])

    def test_cancel_does_not_send_partial_baseline(self):
        with patch('luban.growth_service.run_growth',side_effect=InterruptedError):
            self.service.start();self.service.worker.join(3)
        self.assertEqual(self.service.state['state'],'cancelled')
        self.assertTrue(all(p.endswith('/challenge') for p,_ in self.calls))
        self.assertEqual(self.runtime.state,'ready')

    def test_invalid_server_arithmetic_or_version_rejected(self):
        for change in ({'net_delta':99},{'wrong_to_correct':24},{'eligible':False},{'model_version':A},
                       {'baseline_score':float('nan')},{'question_count':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validated_growth({**RECEIPT,**change})

    def test_learning_import_and_teacher_capture_exclude_reserved_tasks(self):
        self.run_job()
        prompt='Reserved fresh task 0'
        with self.assertRaisesRegex(ValueError,'只能检验'):
            self.memory.add_sample(prompt,'1',approved=True,training_allowed=True)
        self.memory.capture_teacher([{'role':'user','content':prompt}],'1',{},True)
        self.assertEqual(self.memory.samples(),[])
        self.assertEqual(self.memory.teacher_materials()[0]['status'],'pending')
        # Imported/directly copied records are excluded at the common sample accessor too.
        with self.memory.lock,self.memory.connect() as db:
            db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)',('copied',prompt,'1','human','training',time.time(),1,1,'{}'))
        self.assertEqual(self.memory.samples(),[])

    def test_private_original_retests_not_sent_to_server(self):
        with patch('luban.growth_service.eligible_cases',return_value=[{'prompt':'My private pet question','answer':'private'}]):
            self.run_job()
        self.assertNotIn('My private pet question',json.dumps(self.calls))
        self.assertEqual(self.service._read()['active']['cases'][0]['prompt'],'My private pet question')

    def test_growth_read_rejects_cross_domain_response(self):
        def get(request):
            return httpx.Response(200,json={'protocol_version':2,'measurement':'accepted_branch_upgrade_records',
                'sort':'growth','domain':'programming','entries':[],'total':0,'page':1,'page_size':100})
        self.board.transport=httpx.MockTransport(get)
        self.board.data['country_detected']=True
        result=self.board.read(1,'growth','mathematics')
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['entries'],[])

    def enable(self):
        data=self.service._read();data['enabled']=True;self.service._save(data)

    def test_pause_before_worker_entry_cannot_be_cleared_by_budget_start(self):
        worker=Mock()
        worker.is_alive.return_value=False
        with patch('luban.growth_service.threading.Thread',return_value=worker):
            self.service.start()
            self.service.pause()
        self.service._work()
        self.assertEqual(self.service.state['state'],'cancelled')
        self.assertEqual(self.calls,[])
        self.assertEqual(self.runtime.state,'ready')

    def test_late_cancelled_timer_callback_cannot_cancel_the_next_round(self):
        timers=[]
        def timer(seconds,callback):
            value=Mock();value.callback=callback;timers.append(value)
            self.assertEqual(seconds,300)
            return value
        with patch('luban.growth_service.threading.Timer',side_effect=timer):
            self.service._begin_budget();old=self.service.stop_event
            self.service._end_budget()
            self.service.stop_event=threading.Event()
            self.service._begin_budget()
            timers[0].callback()
            self.assertTrue(old.is_set())
            self.assertFalse(self.service.stop_event.is_set())
            self.service._check_budget()
            self.service._end_budget()
        self.assertTrue(all(timer.cancel.called for timer in timers))

    def test_pause_after_generation_before_upload_discards_partial_baseline(self):
        def late_pause(*args,**kwargs):
            report=self.generate(*args,**kwargs)
            self.service.stop_event.set()
            return report
        with patch('luban.growth_service.run_growth',side_effect=late_pause):
            self.service.start();self.service.worker.join(3)
        self.assertEqual(self.service.state['state'],'cancelled')
        self.assertIsNone(self.service._read()['active'])
        self.assertTrue(all(path.endswith('/challenge') for path,_ in self.calls))

    def test_300_second_timeout_keeps_existing_baseline_and_freezes_elapsed(self):
        self.run_job();self.version=B
        def over_budget(*args,**kwargs):
            self.assertAlmostEqual(kwargs['deadline_monotonic'],self.service.started_at+300)
            report=self.generate(*args,**kwargs)
            self.service.started_at=time.monotonic()-301
            return report
        with patch('luban.growth_service.run_growth',side_effect=over_budget):
            self.service.start();self.service.worker.join(3)
        self.assertEqual(self.service.state['state'],'error')
        self.assertIn('5 分钟',self.service.state['error'])
        self.assertIsNotNone(self.service._read()['active'])
        self.assertFalse(any(path.endswith('/after') for path,_ in self.calls))
        elapsed=self.service.status()['elapsed_seconds']
        with patch('luban.growth_service.time.monotonic',return_value=self.service.started_at+900):
            self.assertEqual(self.service.status()['elapsed_seconds'],elapsed)
        self.assertEqual(self.service.status()['budget_seconds'],300)

    def test_before_training_timer_failure_continues_and_restores_session(self):
        self.enable()
        session=SimpleNamespace(body={'checkpoint':None},pause_requested=False)
        self.runtime.training_session=session
        timer=Mock();timer.start.side_effect=OSError('timer unavailable')
        with patch('luban.growth_service.threading.Timer',return_value=timer):
            self.service.before_training(session)
        self.assertIs(self.runtime.training_session,session)
        self.assertEqual(self.service.state['state'],'error')
        self.assertIn('本轮学习继续',self.service.state['phase'])
        timer.cancel.assert_called_once()
        self.assertEqual(self.calls,[])

    def test_before_training_network_failure_continues_and_restores_session(self):
        self.enable()
        session=SimpleNamespace(body={'checkpoint':None},pause_requested=False)
        self.runtime.training_session=session
        with patch.object(self.board,'_request',side_effect=OSError('offline')):
            self.service.before_training(session)
        self.assertIs(self.runtime.training_session,session)
        self.assertEqual(self.service.state['state'],'error')

    def test_before_training_pause_during_timer_start_remains_a_training_pause(self):
        from luban.training_checkpoint import TrainingPaused
        self.enable()
        session=SimpleNamespace(body={'checkpoint':None},pause_requested=False)
        self.runtime.training_session=session
        timer=Mock();timer.start.side_effect=lambda:self.service.pause()
        with patch('luban.growth_service.threading.Timer',return_value=timer),self.assertRaises(TrainingPaused):
            self.service.before_training(session)
        self.assertIs(self.runtime.training_session,session)
        self.assertEqual(self.calls,[])
        self.assertTrue(session.pause_requested)

    def test_pause_racing_timeout_does_not_continue_detached_training(self):
        from luban.training_checkpoint import TrainingPaused
        self.enable()
        session=SimpleNamespace(body={'checkpoint':None},pause_requested=False)
        self.runtime.training_session=session
        def timeout_and_pause(*args,**kwargs):
            self.service.pause()
            self.service.started_at=time.monotonic()-301
            raise TimeoutError('deadline')
        with patch.object(self.service,'_compare',side_effect=timeout_and_pause),self.assertRaises(TrainingPaused):
            self.service.before_training(session)
        self.assertIs(self.runtime.training_session,session)
        self.assertTrue(session.pause_requested)
        self.assertIsNone(self.service.pretraining_session)

    def test_idle_failure_is_attempted_once_even_after_service_restart(self):
        self.run_job();self.version=B
        with patch.object(self.board,'_request',side_effect=OSError('offline')) as request:
            self.assertTrue(self.service.idle())
            self.service.worker.join(3)
            self.assertEqual(self.service.state['state'],'error')
            for _ in range(3):self.assertFalse(self.service.idle())
            recreated=GrowthService(self.runtime,self.board,threading.Lock(),lambda:False)
            self.assertFalse(recreated.idle())
            request.assert_called_once()
            recreated.close()
        # Explicit user retry remains available; automatic failure did not erase the baseline.
        self.run_job()
        self.assertEqual(self.service.state['state'],'complete')

    def test_pause_invalidates_deferred_after_upgrade_waiter(self):
        self.enable()
        targets=[]
        def thread(**kwargs):
            targets.append(kwargs['target'])
            return Mock()
        with patch('luban.growth_service.threading.Thread',side_effect=thread),patch.object(self.service,'idle') as idle:
            self.service.after_upgrade()
            self.service.pause()
            targets[0]()
            idle.assert_not_called()

    def test_delayed_retention_accepts_flat_protocol_and_preserves_receipt(self):
        self.run_job();self.version=B;self.run_job()
        data=self.service._read()
        data['history'][0]['receipt']['completed_at']=(NOW-timedelta(days=2)).isoformat()
        self.service._save(data)
        self.run_job()
        receipt=self.service._read()['history'][0]['receipt']
        self.assertEqual(receipt['retention_check']['retained_correct'],17)
        self.assertEqual(receipt['retention_check']['retention'],94.44)
        self.assertEqual(receipt['net_delta'],25)
        self.assertEqual(sum(path.endswith('/retention') for path,_ in self.calls),1)
        self.run_job()
        self.assertEqual(sum(path.endswith('/retention') for path,_ in self.calls),1)

    def test_old_policy_active_expires_without_backfilled_before_or_unreserving(self):
        self.run_job()
        data=self.service._read();reserved=data['reserved_prompts'][:]
        data['active']['challenge']['policy_hash']='9'*64;self.service._save(data)
        self.version=B;self.calls.clear();self.run_job()
        self.assertEqual(self.service.state['state'],'expired')
        self.assertEqual(self.calls,[])
        data=self.service._read()
        self.assertIsNone(data['active'])
        self.assertEqual(data['reserved_prompts'],reserved)

    def test_old_policy_receipt_is_archived_but_not_profile_or_publication(self):
        self.run_job();self.version=B;self.run_job()
        data=self.service._read();data['history'][0]['receipt']['policy_hash']='9'*64
        self.service._save(data)
        self.assertIsNone(self.service.profile()['growth'])
        self.assertIsNone(self.service.publication())
        self.assertFalse(self.service.idle())
        self.assertEqual(len(self.service._read()['history']),1)
        with self.assertRaisesRegex(ValueError,'条件已变化'):
            validated_growth(data['history'][0]['receipt'])

    def test_private_original_shares_round_deadline(self):
        cases=[{'prompt':'My private question','answer':'private'}]
        with patch('luban.growth_service.eligible_cases',return_value=cases), \
                patch('luban.growth_service.original_retest',return_value={}) as original:
            self.run_job()
        self.assertAlmostEqual(original.call_args.kwargs['deadline_monotonic'],self.service.started_at+300)

    def test_thread_start_failure_restores_runtime_without_a_phantom_worker(self):
        worker=Mock();worker.start.side_effect=OSError('thread unavailable')
        with patch('luban.growth_service.threading.Thread',return_value=worker),self.assertRaises(OSError):
            self.service.start()
        self.assertEqual(self.runtime.state,'ready')
        self.assertIsNone(self.service.worker)


if __name__=='__main__':unittest.main()
