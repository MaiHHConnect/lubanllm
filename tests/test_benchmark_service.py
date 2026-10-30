"""Job publication uses server receipts; no paid API or model training here."""
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from luban.benchmark import POLICY_HASH, SUITE_ID, SUITE_SHA256
from luban.benchmark_service import BenchmarkService, validated_receipt
from luban.leaderboard import Leaderboard
from luban.memory import Memory
from luban.upgrades import read_upgrade_records


VERSION = 'a' * 64
RECEIPT = {'receipt_id': 'b' * 64, 'model_version': VERSION, 'suite_id': SUITE_ID,
           'suite_sha256': SUITE_SHA256, 'policy_hash': POLICY_HASH, 'benchmark': 62.5,
           'generalization': 75, 'benchmark_delta': None, 'generalization_delta': None}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.requests = []
        self.runtime = SimpleNamespace(state='ready', model=object(), job=None,
            memory=Memory(self.root/'storage'), lock=threading.RLock(), _ready=lambda:None)
        self.board = Leaderboard(self.root/'home', storage_root=self.root/'storage',
            transport=httpx.MockTransport(self.remote), country_transport=httpx.MockTransport(lambda r: httpx.Response(500)),
            token_provider=lambda: sum(r['total_tokens'] for r in self.runtime.memory.usage().values()))
        self.board.save_profile({'username': 'Tester', 'country_code': 'JP'})
        self.service = BenchmarkService(self.runtime, self.board, self.root/'home', threading.Lock(), lambda: False)
        self.board.evaluation_provider = self.service.profile
        self.addCleanup(self.service.close)
        self.version = patch('luban.benchmark_service.model_version', return_value=VERSION)
        self.version_mock = self.version.start(); self.addCleanup(self.version.stop)

    def remote(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path.endswith('/challenge'):
            return httpx.Response(200, json={'receipt': RECEIPT})
        if request.url.path.endswith('/evaluation'):
            return httpx.Response(200, json={'receipt': RECEIPT})
        return httpx.Response(200, json={'protocol_version': 2, 'entry': {'username': body['username'],
            'country_code': body['country_code'], 'evolution_count': 0, 'rank': 1,
            'token_total': body.get('token_total'), **{k:RECEIPT[k] for k in ('benchmark','generalization','benchmark_delta','generalization_delta','suite_id','model_version')}}})

    def wait(self):
        self.service.worker.join(3)
        self.assertFalse(self.service.worker.is_alive())

    def test_cached_receipt_publishes_once_without_generating_or_faking_baseline(self):
        self.service.start(); self.wait()
        result = self.service.status()
        self.assertEqual(result['state'], 'complete'); self.assertEqual(result['completed'], 48)
        self.assertIsNone(result['result']['benchmark_delta'])
        posted = self.requests[-1][1]
        self.assertEqual(posted['evaluation_receipt_id'], RECEIPT['receipt_id'])
        self.assertNotIn('benchmark', posted)
        self.assertEqual(self.board.profile()['benchmark'], 62.5)
        self.assertEqual(self.runtime.state, 'ready')
        self.assertEqual(self.runtime.memory.usage()['local']['calls'], 0)

    def test_answer_generation_counts_real_tokens_once_then_submits_complete_answer_batch(self):
        original = self.remote
        def remote(request):
            if request.url.path.endswith('/challenge'):
                return httpx.Response(200, json={'challenge_id':'d'*64, 'suite_sha256':SUITE_SHA256})
            return original(request)
        self.board.transport = httpx.MockTransport(remote)
        report = {'challenge_id':'d'*64, 'model_version':VERSION, 'answers':[{'id':'q','text':'A','finish_reason':'stop'}],
                  'metadata':{'elapsed_seconds':3, 'items':[{'prompt_tokens':11,'completion_tokens':2}]}}
        with patch('luban.benchmark_service.run_benchmark', return_value=report):
            self.service.start(); self.wait()
        self.assertEqual(self.service.status()['state'], 'complete')
        self.assertEqual(self.board.token_total(), 13)
        self.assertEqual(self.runtime.memory.usage()['local']['calls'], 1)
        self.assertEqual(self.requests[-1][1]['token_total'], 13)
        self.assertEqual(len(self.runtime.memory.samples()), 0)

    def test_cancel_does_not_submit_partial_answers_or_start_two_workers(self):
        started = threading.Event()
        def remote(request):return httpx.Response(200,json={'challenge_id':'d'*64,'suite_sha256':SUITE_SHA256})
        self.board.transport = httpx.MockTransport(remote)
        def run(runtime, challenge, stopped, progress, **kwargs):
            started.set(); stopped.wait(2); raise InterruptedError('cancelled')
        with patch('luban.benchmark_service.run_benchmark', side_effect=run):
            self.service.start(); self.assertTrue(started.wait(1))
            with self.assertRaises(RuntimeError):self.service.start()
            self.service.pause()
        self.assertEqual(self.runtime.state, 'ready')
        self.assertEqual(self.service.status()['state'], 'error')
        self.assertFalse(self.service.store.path.exists())

    def test_expired_whole_round_cannot_publish_even_a_cached_valid_receipt(self):
        clock = [100.0]
        original = self.remote
        def remote(request):
            response = original(request)
            clock[0] = 401.0
            return response
        self.board.transport = httpx.MockTransport(remote)
        with patch('luban.benchmark_service.time.monotonic', side_effect=lambda:clock[0]):
            self.service._work()
        self.assertEqual(self.service.status()['state'], 'error')
        self.assertIn('5分钟', self.service.status()['error'])
        self.assertFalse(self.service.store.path.exists())
        self.assertFalse(any(path.endswith('/me') for path, body in self.requests))

    def test_invalid_receipts_and_restored_version_do_not_show_old_current_score(self):
        for key, value in [('benchmark',True),('generalization',101),('benchmark_delta',float('nan')),('policy_hash','f'*64),('suite_sha256','f'*64),('model_version','f'*64)]:
            with self.subTest(key=key), self.assertRaises(ValueError):validated_receipt({**RECEIPT,key:value},VERSION)
        self.service.state['result'] = RECEIPT.copy()
        self.version_mock.return_value = 'f'*64
        self.assertIsNone(self.service.profile()['benchmark'])


class CoreCountTests(unittest.TestCase):
    def test_only_committed_core_product_with_matching_report_weights_and_protection_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); memory=Memory(root); store=root/'modules/luban0-v1'; store.mkdir(parents=True)
            run='0123456789ab'; version='core-'+run
            weights=store/(version+'.safetensors');weights.write_bytes(b'fixture immutable checkpoint')
            retained=store/(version+'.retention.json');retained.write_text('{}')
            meta={'version':version,'identity':VERSION,'report':'evolution-'+run+'.json',
                  'sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                  'retention_sha256':hashlib.sha256(retained.read_bytes()).hexdigest()}
            (store/(version+'.json')).write_text(json.dumps(meta))
            report={'id':run,'version':version,'identity':VERSION,'promoted':True,'publication_state':'published',
                    'checkpoint':meta,'gates':{str(i):True for i in range(7)},'baseline_new_loss':2,
                    'validation_new_loss':1,'baseline_replay_loss':1,'validation_replay_loss':1,
                    'started':time.time()-2,'elapsed_seconds':1}
            (store/('evolution-'+run+'.json')).write_text(json.dumps(report))
            body={'scope':'product','binding':{'identity':VERSION},'commit':{'report':report},
                  'records':{'groups':{'new_training':[{'id':'t'}], 'new_validation':[{'id':'v'}]}}}
            with memory.connect() as db:
                db.execute('CREATE TABLE training_runs(id TEXT,kind TEXT,state TEXT,body TEXT)')
                db.execute('INSERT INTO training_runs VALUES (?,?,?,?)',(run,'evolution','committed',json.dumps(body)))
            receipts=read_upgrade_records(root)
            self.assertEqual(len(receipts),1);self.assertEqual(receipts[0]['kind'],'core')
            body['scope']='research'
            with memory.connect() as db:db.execute('UPDATE training_runs SET body=?',(json.dumps(body),))
            self.assertEqual(read_upgrade_records(root),[])
            body['product_accepted']=True
            with memory.connect() as db:db.execute('UPDATE training_runs SET body=?',(json.dumps(body),))
            weights.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):read_upgrade_records(root)


if __name__ == '__main__':unittest.main()
