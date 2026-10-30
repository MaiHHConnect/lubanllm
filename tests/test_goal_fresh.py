import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from luban import goal_fresh as fresh,evolution

def fixture(passed=1, old89=True):
 groups={k:[{'id':k+str(i)} for i in range(n)] for k,n in [('new_training',24),('new_validation',8),('replay_training',83),('replay_validation',8)]}
 old=[{'id':'old'+str(i)} for i in range(89)];ref=torch.zeros(2,4,dtype=torch.bfloat16);parameter=torch.nn.Parameter(ref.clone());master=torch.nn.Parameter(ref.float().clone());optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
 for _ in range(450):optimizer.zero_grad();master.square().sum().backward();optimizer.step()
 protocol={'protocol':fresh.PROTOCOL,'_protocol_sha256':'a'*64,'replay_generation_ids':[r['id'] for r in groups['replay_training'][:13]]}
 runtime=SimpleNamespace(branch=SimpleNamespace(expert=None),training_session=SimpleNamespace(body={'records':{'protection':old}}),job={})
 calls=[]
 def generation(runtime, rows):
  calls.extend(r['id'] for r in rows)
  return [{'id':r['id'],'passed':(i<passed if r['id'].startswith('new') else old89 if r['id'].startswith('old') else True),'complete':True,'branch_enabled':False,'reference_used':False} for i,r in enumerate(rows)]
 selection={'checkpoints':fresh.CHECKPOINTS,'policy_version':fresh.POLICY,'required_consecutive_passes':0,'required_same_weight_passes':2,'required_new_exposures':2,'replay_generation_ids':protocol['replay_generation_ids'],'minimum_replay_generation':4,'results':[],'selected_step':None}
 report={'id':'abcdef123456','version':'core-abcdef123456','steps':2700,'dataset_sha256':'d','parameter':'weight','split_ids':{k:[r['id'] for r in v] for k,v in groups.items()},'training_engine':{'engine':'cpu-test','binary_sha256':'b'},'goal_sequence':fresh.report_metadata(protocol),'training_selection':selection,'baseline_training_replay_loss':1.0}
 # Reference is genuinely the same toy weight; proof generator is a controller mock.
 report['goal_sequence']['reference_baseline_before_training']=fresh.observe(runtime,protocol,groups,parameter,0,generation,lambda *x:1.0)
 return runtime,protocol,groups,parameter,master,optimizer,report,calls,generation

class FreshTests(unittest.TestCase):
 def test_boundary_partial_can_carry_and_never_reads_new_validation(self):
  r,p,g,w,m,o,report,calls,gen=fixture();proof=fresh.observe_boundary(r,p,report,g,w,450,gen,lambda *x:1.)
  self.assertEqual(proof['round_result'],'partial');self.assertEqual(proof['pending_count'],23);self.assertIsNone(report['training_selection']['selected_step'])
  self.assertTrue(proof['old89_both_passed']);self.assertEqual(len([c for c in calls if c.startswith('old')]),356)
  self.assertFalse(any(c.startswith('new_validation') for c in calls))
 def test_satisfaction_at_first_complete450_enters_final_selection(self):
  r,p,g,w,m,o,report,calls,gen=fixture(passed=24);proof=fresh.observe_boundary(r,p,report,g,w,450,gen,lambda *x:1.)
  self.assertEqual(report['training_selection']['selected_step'],450);self.assertEqual(proof['actual_updates'],450)
  with self.assertRaises(ValueError):fresh.observe_boundary(r,p,report,g,w,900,gen,lambda *x:1.)
 def test_full89_failure_stops_even_when_old13_and_average_loss_pass(self):
  r,p,g,w,m,o,report,calls,gen=fixture();report['goal_sequence']['reference_baseline_before_training']['old89_both_passed']=True
  def failed(runtime,rows):
   out=gen(runtime,rows)
   if out and out[0]['id'].startswith('old'):out[0]['passed']=False
   return out
  proof=fresh.observe_boundary(r,p,report,g,w,450,failed,lambda *x:1.)
  self.assertTrue(proof['old13_both_passed']);self.assertTrue(proof['replay_loss_both_passed']);self.assertTrue(fresh.stopped(report));self.assertEqual(proof['round_result'],'protection_failed')
 def test_no_448_or_skipped_boundaries(self):
  r,p,g,w,m,o,report,calls,gen=fixture()
  for step in (448,868,900):
   with self.assertRaises(ValueError):fresh.observe_boundary(r,p,report,g,w,step,gen,lambda *x:1.)
 def test_real_resume_restores_complete_adam_master_and_rejects_tampering(self):
  r,p,g,w,m,o,report,calls,gen=fixture();fresh.observe_boundary(r,p,report,g,w,450,gen,lambda *x:1.)
  with tempfile.TemporaryDirectory() as d:
   cap={'records':r.training_session.body['records']['protection']};path=Path(d).resolve()/'capture.json';path.write_text(json.dumps(cap));basis=torch.eye(4)[:,:1]
   sources={'original_tensor_sha256':fresh.tensor_sha(w),'protection_capture':{'path':str(path),'sha256':fresh.common.file_sha(path)}}
   report['goal_sequence_sources']=sources;report['replay_subspace']={'protection_capture_sha256':fresh.content_sha(cap),'basis_sha256':fresh.tensor_sha(basis)}
   payload={'format':1,'id':report['id'],'step':450,'original_sha256':fresh.tensor_sha(w),'dataset_sha256':'d','parameter':'weight','master':m.detach().clone(),'basis':basis,'optimizer':o.state_dict(),'new_exposures':{s['id']:225//24+int(i<225%24) for i,s in enumerate(g['new_training'])},'consecutive_passes':0,'report':report}
   fresh.validate_resume_history(payload,copy.deepcopy(report),g)
   target_master=torch.nn.Parameter(torch.ones_like(m));target_optimizer=torch.optim.Adam([target_master],lr=1e-5,eps=1e-6)
   restored=evolution._resume_core(SimpleNamespace(id=report['id']),payload,copy.deepcopy(report),w,w.detach().clone(),g,2700,target_master,target_optimizer)
   self.assertTrue(torch.equal(target_master,payload['master']));self.assertEqual(float(next(iter(target_optimizer.state.values()))['step']),450);self.assertEqual(restored['goal_sequence'],report['goal_sequence'])
   edits=[lambda q:q['optimizer']['state'][0]['step'].fill_(449),lambda q:q['basis'].add_(1),lambda q:q['new_exposures'].update(new_training0=0),lambda q:q['report']['goal_sequence']['stage_observations'][0].update(pending_count=0),lambda q:q['report']['goal_sequence']['stage_observations'][0]['observations'][0]['old89'][0].update(id='new_validation0')]
   for edit in edits:
    bad=copy.deepcopy(payload);edit(bad)
    with self.assertRaises(ValueError):fresh.validate_resume_history(bad,copy.deepcopy(report),g)
 def test_new_objective_never_imports_failed_parent(self):self.assertIsNone(fresh.initial_saved())
if __name__=='__main__':unittest.main()
