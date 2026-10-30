import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from luban import goal_fresh as f
from test_goal_fresh import fixture

def closed_fixture():
 r,p,g,w,m,o,report,calls,gen=fixture();report.update(started=1.,promoted=False,partial_policy=None)
 # CPU controller fixture, not a real2700/model run.
 for step in f.CHECKPOINTS:f.observe_boundary(r,p,report,g,w,step,gen,lambda *x:1.)
 for _ in range(2250):o.zero_grad();m.square().sum().backward();o.step()
 temp=tempfile.TemporaryDirectory();cap={'records':r.training_session.body['records']['protection']}
 path=Path(temp.name).resolve()/'capture.json';path.write_text(json.dumps(cap));basis=torch.eye(4)[:,:1]
 report['goal_sequence_sources']={'protocol_sha256':'p'*64,'original_tensor_sha256':f.tensor_sha(w),'protection_capture':{'path':str(path),'sha256':f.common.file_sha(path)}}
 report['replay_subspace']={'protection_capture_sha256':f.content_sha(cap),'basis_sha256':f.tensor_sha(basis)}
 report['training_engine'].update(segments=[])
 records={'groups':g,'protection':r.training_session.body['records']['protection'],'retention_regressions':[]}
 binding={'arguments':f.arguments(p),'implementation_sha256':{'old':'a'},'records_sha256':f.content_sha(records),'scope':'product','core_version':'B45'}
 payload={'format':1,'id':'abcdef123456','parameter':'weight','dataset_sha256':'d','original_sha256':f.tensor_sha(w),'step':2700,'master':m.detach().clone(),'basis':basis,'optimizer':o.state_dict(),'new_exposures':{v['id']:1350//24+int(i<1350%24) for i,v in enumerate(g['new_training'])},'kl_total':0.,'kl_maximum':0.,'consecutive_passes':0,'report':report}
 closed={'binding':binding,'payload':payload,'rng':torch.get_rng_state().clone(),'mps_rng':torch.get_rng_state().clone()}
 proto={**p,'protocol':f.RESUME_PROTOCOL,'parent_run_id':payload['id'],'new_run_id':'987654abcdef','parent_steps':2700,'total_steps':4050,'_parent_report':report,'_parent_binding':binding,'files':{},'parent_binding_sha256':f.content_sha(binding),'parent_report_sha256':f.content_sha(report),'parent_history_sha256':f.content_sha(report['training_selection']['results']),'parent_source_sha256':{},'parent_implementation_sha256':binding['implementation_sha256'],'new_implementation_sha256':{'new':'b'},'new_source_sha256':{},'basis_sha256':f.tensor_sha(basis),'reference_sha256':f.tensor_sha(w.float()),'parent_final_tensor_sha256':f.tensor_sha(w),'original_tensor_sha256':f.tensor_sha(w),'parameter':'weight','dataset_sha256':'d','records_sha256':f.content_sha(records)}
 new_binding={**binding,'arguments':f.arguments(proto),'implementation_sha256':proto['new_implementation_sha256']}
 native={'candidate':w.detach().clone(),'master':m.detach().clone(),'exp_avg':o.state_dict()['state'][0]['exp_avg'].clone(),'exp_avg_sq':o.state_dict()['state'][0]['exp_avg_sq'].clone(),'adam_step':torch.tensor(2700.),'reference':w.float(),'basis':basis.clone()}
 return temp,r,proto,g,closed,native,new_binding,records,w,gen

class ContinuationTests(unittest.TestCase):
 def test_no_parent_closure_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d).resolve()/'protocol.json';path.write_text(json.dumps({'protocol':f.RESUME_PROTOCOL,'status':'PRE_REGISTERED_NOT_TRAINED'}))
   with self.assertRaises(ValueError):f.load_verified(path)
 def test_fullstate_derivation_preserves_six_rounds_and_all_tensors(self):
  temp,r,p,g,c,n,b,records,w,gen=closed_fixture()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   old=copy.deepcopy(c);q=f.derive_payload(p,c,n,b,records)
   self.assertEqual(q['step'],2700);self.assertEqual(q['report']['steps'],4050);self.assertEqual(q['new_exposures'],old['payload']['new_exposures'])
   for key in ('master','basis'):self.assertTrue(torch.equal(q[key],old['payload'][key]))
   for key in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(q['optimizer']['state'][0][key],old['payload']['optimizer']['state'][0][key]))
   self.assertTrue(torch.equal(c['rng'],old['rng']));self.assertTrue(torch.equal(c['mps_rng'],old['mps_rng']))
   self.assertEqual(q['report']['goal_sequence']['stage_observations'],old['payload']['report']['goal_sequence']['stage_observations'])
   self.assertEqual(q['report']['training_selection']['results'],old['payload']['report']['training_selection']['results'])
   f.validate_resume_history(q,copy.deepcopy(q['report']),g)
 def test_parent_state_or_rng_or_binding_tamper_fails(self):
  temp,r,p,g,c,n,b,records,w,gen=closed_fixture()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   for edit in (lambda x:x['payload']['master'].add_(1),lambda x:x['payload']['optimizer']['state'][0]['step'].fill_(0),lambda x:x.update(mps_rng=None),lambda x:x['payload']['report']['training_selection'].update(selected_step=2700)):
    bad=copy.deepcopy(c);edit(bad)
    with self.assertRaises(ValueError):f.derive_payload(p,bad,n,b,records)
   wrong={**b,'core_version':'other'}
   with self.assertRaises(ValueError):f.derive_payload(p,c,n,wrong,records)
 def test_next_complete450_preserves_prior_mastery_and_never_reads_valid(self):
  temp,r,p,g,c,n,b,records,w,gen=closed_fixture()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=f.derive_payload(p,c,n,b,records);r.training_session.body['records']=records
   proof=f.observe_boundary(r,p,q['report'],g,w,3150,gen,lambda *x:1.)
   self.assertEqual(proof['pending_count'],23);self.assertEqual(proof['actual_updates'],450);self.assertEqual(q['report']['goal_sequence']['new_steps_executed'],450)
   self.assertEqual(len(q['report']['goal_sequence']['stage_observations']),7)
   for step in (3600,4050):f.observe_boundary(r,p,q['report'],g,w,step,gen,lambda *x:1.)
   self.assertEqual(len(q['report']['goal_sequence']['stage_observations']),9)
   with self.assertRaises(ValueError):f.observe_boundary(r,p,q['report'],g,w,4500,gen,lambda *x:1.)
 def test_adam_eps_real_value_is_required(self):
  temp,r,p,g,c,n,b,records,w,gen=closed_fixture()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   c['payload']['optimizer']['param_groups'][0]['eps']=1e-8
   with self.assertRaises(ValueError):f.derive_payload(p,c,n,b,records)

if __name__=='__main__':unittest.main()
