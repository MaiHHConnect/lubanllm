"""CPU tiny state: real helper, real Adam/resume; no backbone/native worker/GPU."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from luban import goal_sequence as g
from luban.evolution import _resume_core, _consolidate_loaded
import test_goal_sequence as oldtests


class NextThreeRoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        oldtests.GoalSequenceTests.setUpClass()

    def setUp(self):
        self.shape=patch.object(g,'SHAPE',(2,4));self.shape.start();self.addCleanup(self.shape.stop)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        original=oldtests.GoalSequenceTests();original.values=copy.deepcopy(oldtests.GoalSequenceTests.base)
        runtime, parameter, saved=original.round_fixture()
        p,closed,native,binding,records,closure=original.values
        master=torch.nn.Parameter(saved['master'].clone());opt=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
        opt.load_state_dict(saved['optimizer'])
        for step, count in ((900,2),(1350,11),(1800,15)):
            for _ in range(450):
                opt.zero_grad(set_to_none=True);master.grad=torch.full_like(master,.07);opt.step()
                with torch.no_grad():master[:,0]=0
            with torch.no_grad():parameter.copy_(master.to(torch.bfloat16))
            original.run_boundary(runtime,parameter,saved,step,['new'+str(i) for i in range(count)])
        saved.update(step=1800, master=master.detach().clone(), optimizer=copy.deepcopy(opt.state_dict()),
            new_exposures={'new'+str(i):56+int(i<4) for i in range(16)})
        saved['report'].update(completed_steps=1800, elapsed_seconds=1.)
        native.update(master=saved['master'].clone(),candidate=saved['master'].to(torch.bfloat16),
            exp_avg=opt.state[master]['exp_avg'].clone(),exp_avg_sq=opt.state[master]['exp_avg_sq'].clone(),
            adam_step=opt.state[master]['step'].clone())
        ancestor_report=self.root/'ancestor-report.json';ancestor_report.write_text(json.dumps(closed['payload']['report']))
        p['files']['parent_report']={'path':str(ancestor_report),'sha256':g.file_sha(ancestor_report)}
        parent_binding=copy.deepcopy(binding)
        envelope={'binding':parent_binding,'payload':saved,'rng':closed['rng'].clone(),'mps_rng':closed['mps_rng'].clone()}
        reportpath=self.root/'parent-report.json';reportpath.write_text(json.dumps(saved['report']))
        protocol=copy.deepcopy(p)
        protocol.update(protocol=g.PROTOCOL_V3,parent_steps=1800,max_steps=3150,max_segments=7,
            checkpoints=g.CHECKPOINTS_V3.copy(), parent_run_id=saved['id'],new_run_id='999999abcdef',
            parent_arguments=g.arguments(p),parent_sequence_metadata=g.metadata(p),
            parent_binding_sha256=g.content_sha(parent_binding), parent_report_sha256=g.content_sha(saved['report']),
            parent_final_tensor_sha256=g.tensor_sha(native['candidate']),
            parent_implementation_sha256=parent_binding['implementation_sha256'],
            new_implementation_sha256={**parent_binding['implementation_sha256'],'goal_sequence.py':'9'*64},
            inherited_observations_sha256=g.content_sha(g.inherited_observations(saved['report'])),
            parent_goal_sequence_sources_sha256=g.content_sha(saved['report']['goal_sequence_sources']),
            _protocol_sha256='3'*64)
        protocol['files']['parent_report']={'path':str(reportpath),'sha256':g.file_sha(reportpath)}
        protocol['files']['parent_sequence_protocol']={'path':str(self.root/'ancestor.json'),'sha256':p['_protocol_sha256']}
        protocol['closure']['final_step']=1800
        newbinding=copy.deepcopy(parent_binding);newbinding.update(arguments=g.arguments(protocol),implementation_sha256=protocol['new_implementation_sha256'])
        self.values=protocol,envelope,native,newbinding,records,protocol['closure']
        self.ancestor=p;self.runtime=runtime;self.parameter=parameter
        self.loader=patch.object(g,'load_verified',return_value=p);self.loader.start();self.addCleanup(self.loader.stop)

    def derive(self):return g.derive_payload(*self.values)

    def enter(self):
        p,_,native,_,records,_=self.values
        payload=self.derive()
        def generation(runtime,rows):
            return [{'id':r['id'],'passed':r['id'].startswith('oldtrain') or r['id'] in {'new'+str(i) for i in range(15)},
                'complete':True,'branch_enabled':False,'reference_used':False} for r in rows]
        payload['report']['goal_sequence'].update(g.entry_observations(self.runtime,p,payload,self.parameter,
            native['reference'].to(torch.bfloat16),records['groups'],generation,lambda *_:1.))
        g.classify_round(payload['report']['goal_sequence']['entry_parent'],1.06)
        return payload

    def boundary(self,payload,step,count=15,old=True,loss=1.):
        p,_,_,_,records,_=self.values
        def gen(runtime,rows):
            return [{'id':r['id'],'passed':old if r['id'].startswith('oldtrain') else r['id'] in {'new'+str(i) for i in range(count)},
                'complete':True,'branch_enabled':False,'reference_used':False} for r in rows]
        return g.observe_boundary(self.runtime,p,payload['report'],records['groups'],self.parameter,step,gen,lambda *_:loss)

    def test_complete1800_adam_and_original_history_into_new3150_session(self):
        p,closed,native,binding,records,closure=self.values;before=copy.deepcopy(closed)
        derived=self.derive()
        self.assertEqual(g.metadata(p)['parent_steps'],1800);self.assertEqual(g.arguments(p)['steps'],3150)
        self.assertEqual(derived['id'],'999999abcdef');self.assertEqual(derived['step'],1800)
        self.assertEqual(float(next(iter(derived['optimizer']['state'].values()))['step']),1800)
        self.assertTrue(torch.equal(derived['master'],native['master']))
        self.assertTrue(torch.equal(derived['basis'],native['basis']))
        self.assertEqual(g.inherited_observations(derived['report']),g.inherited_observations(before['payload']['report']))
        self.assertEqual(derived['report']['training_selection']['results'],before['payload']['report']['training_selection']['results'])
        self.assertEqual(derived['new_exposures'],before['payload']['new_exposures'])
        self.assertEqual(closed['payload']['report'],before['payload']['report'])
        self.assertEqual(derived['report']['goal_sequence_sources']['lineage_parent']['parent_final_tensor_sha256'],p['parent_final_tensor_sha256'])

    def test_enter1800_rechecks_actual_same_W_without_new_optimization_or_heldout(self):
        before=self.parameter.clone();payload=self.enter()
        proof=payload['report']['goal_sequence']['entry_parent']
        self.assertEqual((proof['step'],proof['confirmed_count'],proof['pending_count']),(1800,15,1))
        self.assertEqual(proof['forgotten_since_prior'],[]);self.assertEqual(proof['validation_rows_accessed'],0)
        self.assertTrue(torch.equal(before,self.parameter))
        g.validate_resume_history(payload,payload['report'],self.values[4]['groups'])

    def test_partial2250_carries_then_complete2700_selects(self):
        payload=self.enter();self.boundary(payload,2250)
        self.assertIsNone(payload['report']['training_selection']['selected_step'])
        self.assertEqual(payload['report']['goal_sequence']['new_steps_executed'],450)
        self.boundary(payload,2700,16)
        self.assertEqual(payload['report']['training_selection']['selected_step'],2700)
        self.assertEqual([r['step'] for r in payload['report']['goal_sequence']['stage_observations']],[900,1350,1800,2250,2700])
        with self.assertRaises(ValueError):self.boundary(payload,3150,16)

    def test_first_new_complete2250_can_select_without_checkpoint_temporal_streak(self):
        payload=self.enter();self.boundary(payload,2250,16)
        self.assertEqual(payload['report']['training_selection']['selected_step'],2250)
        self.assertEqual(payload['report']['training_selection']['results'][-1]['same_weight_passes'],2)
        self.assertEqual(payload['consecutive_passes'],0)

    def test_no_early_selection_missing_rounds_or_budget_extension(self):
        payload=self.enter()
        for step in (2218,2668,3118,2700,3600):
            with self.subTest(step=step),self.assertRaises(ValueError):self.boundary(payload,step)
        for key,value in (('max_steps',3600),('parent_steps',450),('max_segments',8),('segment_steps',449)):
            altered=copy.deepcopy(self.values[0]);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):g.validate_protocol(altered)
        with self.assertRaisesRegex(ValueError,'64–512'):_consolidate_loaded(SimpleNamespace(),3150,preserve_replay_subspace=True)

    def test_prior15_or_old13_or_loss_failure_stops_without_selection(self):
        for count,old,loss in ((14,True,1.),(16,False,1.),(16,True,1.061)):
            payload=self.enter();proof=self.boundary(payload,2250,count,old,loss)
            self.assertEqual(proof['round_result'],'protection_failed');self.assertTrue(g.stopped(payload['report']))
            self.assertIsNone(payload['report']['training_selection']['selected_step'])
            with self.assertRaises(ValueError):self.boundary(payload,2700,16)

    def test_actual_resume_core_new_ID_and_immutable_full_ledger(self):
        payload=self.enter()
        fullmaster=torch.nn.Parameter(payload['master'].clone());fullopt=torch.optim.Adam([fullmaster],lr=1e-5,eps=1e-6);fullopt.load_state_dict(payload['optimizer'])
        for _ in range(450):
            fullopt.zero_grad(set_to_none=True);fullmaster.grad=torch.full_like(fullmaster,.03);fullopt.step()
            with torch.no_grad():fullmaster[:,0]=0
        payload.update(master=fullmaster.detach().clone(),optimizer=copy.deepcopy(fullopt.state_dict()))
        with torch.no_grad():self.parameter.copy_(fullmaster.to(torch.bfloat16))
        self.boundary(payload,2250)
        payload.update(step=2250,new_exposures={'new'+str(i):70+int(i<5) for i in range(16)})
        declared=copy.deepcopy(payload['report']);master=torch.nn.Parameter(torch.zeros((2,4)))
        opt=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
        _resume_core(SimpleNamespace(id=payload['id']),payload,declared,self.parameter,
            self.values[2]['reference'].to(torch.bfloat16),self.values[4]['groups'],3150,master,opt)
        self.assertEqual(float(next(iter(opt.state.values()))['step']),2250)
        for change in (lambda x:x['report']['goal_sequence']['entry_parent'].update(pending_count=0),
            lambda x:x['report']['goal_sequence']['stage_observations'][2].update(confirmed_count=16),
            lambda x:x['report']['training_selection']['results'][len(g.HISTORY)].update(replay_loss=.5),
            lambda x:x['report']['training_selection'].update(selected_step=2218),lambda x:x.update(id='other'),
            lambda x:next(iter(x['optimizer']['state'].values()))['step'].fill_(1800),
            lambda x:x['new_exposures'].update(new0=0)):
            altered=copy.deepcopy(payload);change(altered)
            with self.assertRaises(ValueError):_resume_core(SimpleNamespace(id=payload['id']),altered,declared,self.parameter,
                self.values[2]['reference'].to(torch.bfloat16),self.values[4]['groups'],3150,master,opt)

    def test_derive_rejects_parent_adam_reset_source_proof_exposure_and_reference_tamper(self):
        for change in (lambda p,c,n,b,r,cl:n['adam_step'].fill_(450),
            lambda p,c,n,b,r,cl:c['payload']['new_exposures'].update(new0=14),
            lambda p,c,n,b,r,cl:n['reference'].fill_(1),
            lambda p,c,n,b,r,cl:p.update(inherited_observations_sha256='0'*64),
            lambda p,c,n,b,r,cl:p['parent_implementation_sha256'].update(**{'evolution.py':'1'*64}),
            lambda p,c,n,b,r,cl:cl.update(validation_reads=1),
            lambda p,c,n,b,r,cl:c['payload']['report']['goal_sequence']['stage_observations'].pop()):
            original=self.values;self.values=copy.deepcopy(original);change(*self.values)
            try:
                with self.assertRaises((ValueError,KeyError)):self.derive()
            finally:self.values=original


if __name__=='__main__':
    unittest.main(verbosity=2)
