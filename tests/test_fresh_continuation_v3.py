import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from luban import goal_fresh as f,evolution
from test_fresh_continuation import closed_fixture

# Everything4050 below is a toy/controller CPU state, never an actual model run.
def fixture4050():
 temp,r,p,g,c,n,b,records,w,gen=closed_fixture()
 with patch.object(f.common,'SHAPE',(2,4)):q=f.derive_payload(p,c,n,b,records)
 for step in (3150,3600,4050):f.observe_boundary(r,p,q['report'],g,w,step,gen,lambda *x:1.)
 master=torch.nn.Parameter(q['master'].clone());optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6);optimizer.load_state_dict(q['optimizer'])
 for _ in range(1350):optimizer.zero_grad();master.square().sum().backward();optimizer.step()
 q.update(step=4050,master=master.detach().clone(),optimizer=optimizer.state_dict(),new_exposures={v['id']:2025//24+int(i<2025%24) for i,v in enumerate(g['new_training'])})
 c2={'binding':b,'payload':q,'rng':c['rng'].clone(),'mps_rng':c['mps_rng'].clone()}
 p3={**p,'protocol':f.RESUME_PROTOCOL_V3,'parent_run_id':q['id'],'new_run_id':'314159265abc','parent_steps':4050,'total_steps':5400,
     '_parent_report':q['report'],'_parent_binding':b,'parent_binding_sha256':f.content_sha(b),'parent_report_sha256':f.content_sha(q['report']),
     'parent_history_sha256':f.content_sha(q['report']['training_selection']['results']),'parent_implementation_sha256':b['implementation_sha256'],
     'new_implementation_sha256':{'v3':'c'},'parent_goal_sequence_sources_sha256':f.content_sha(q['report']['goal_sequence_sources']),
     'ancestor_lineage':copy.deepcopy(q['report']['goal_sequence_sources']['lineage_parent'])}
 b3={**b,'arguments':f.arguments(p3),'implementation_sha256':p3['new_implementation_sha256']}
 n3={**n,'master':master.detach().clone(),'exp_avg':optimizer.state_dict()['state'][0]['exp_avg'].clone(),'exp_avg_sq':optimizer.state_dict()['state'][0]['exp_avg_sq'].clone(),'adam_step':torch.tensor(4050.)}
 return temp,r,p3,g,c2,n3,b3,records,w,gen

class FreshV3Tests(unittest.TestCase):
 def test_actual_v2_wrapper_still_loads_readonly_without_candidate_outputs(self):
  p=Path('/Volumes/移动数据库/lubanllm/experiments/goal-prospective-fixed450-v2-protocol-20261011T072949Z/protocol.json')
  v=f.load_verified(p)
  self.assertEqual(v['new_run_id'],'25bb1aa918d8');self.assertEqual(f.metadata(v)['parent_steps'],2700)
  self.assertEqual(f.source_metadata(v)['scope'],'closed fresh2700 full-state continuation; originalB45 reference/basis; no failed old v3 ancestry')
 def test_running_parent_cannot_pass_v3_preflight(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d).resolve()/'protocol.json'
   # No actual4050 closure exists. Missing immutable evidence must remain fatal.
   path.write_text(json.dumps({'protocol':f.RESUME_PROTOCOL_V3,'status':'FROZEN_CLOSED_PARENT','parent_steps':4050,'total_steps':5400,
    'segment_steps':450,'policy_version':f.POLICY,'parent_run_id':'25bb1aa918d8','new_run_id':'314159265abc','files':{}}))
   with self.assertRaisesRegex(ValueError,'complete actual parent closure'):f.load_verified(path)
 def test_unregistered_budget_types_and_wrong_parent_rejected(self):
  base={'protocol':f.RESUME_PROTOCOL_V3,'status':'FROZEN_CLOSED_PARENT','parent_steps':4050,'total_steps':5400,
    'segment_steps':450,'policy_version':f.POLICY,'parent_run_id':'25bb1aa918d8','new_run_id':'314159265abc','files':{}}
  with tempfile.TemporaryDirectory() as d:
   path=Path(d).resolve()/'protocol.json'
   for edit in ({'parent_steps':True},{'total_steps':5850},{'segment_steps':449},{'parent_run_id':'e3d9ef987253'},{'new_run_id':'25bb1aa918d8'}):
    path.write_text(json.dumps({**base,**edit}))
    with self.assertRaises(ValueError):f.load_verified(path)
 def test_toy4050_derives_and_preserves_all_nine_rounds_and_2700_ancestry(self):
  temp,r,p,g,c,n,b,records,w,gen=fixture4050()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   before=copy.deepcopy(c);q=f.derive_payload(p,c,n,b,records)
   self.assertEqual(q['step'],4050);self.assertEqual(q['report']['steps'],5400)
   self.assertEqual(q['report']['goal_sequence']['stage_observations'],before['payload']['report']['goal_sequence']['stage_observations'])
   self.assertEqual(q['report']['training_selection']['results'],before['payload']['report']['training_selection']['results'])
   for k in ('master','basis'):self.assertTrue(torch.equal(q[k],before['payload'][k]))
   for k in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(q['optimizer']['state'][0][k],before['payload']['optimizer']['state'][0][k]))
   self.assertEqual(q['new_exposures'],before['payload']['new_exposures'])
   self.assertTrue(torch.equal(c['rng'],before['rng']));self.assertTrue(torch.equal(c['mps_rng'],before['mps_rng']))
   lineage=q['report']['goal_sequence_sources']['lineage_parent']
   self.assertEqual(lineage['ancestor_lineage'],before['payload']['report']['goal_sequence_sources']['lineage_parent'])
   self.assertEqual(lineage['parent_goal_sequence_sources_sha256'],f.content_sha(before['payload']['report']['goal_sequence_sources']))
   f.validate_resume_history(q,q['report'],g)
 def test_future_complete450_boundaries_preserve_pending_and_limit(self):
  temp,r,p,g,c,n,b,records,w,gen=fixture4050()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=f.derive_payload(p,c,n,b,records)
   for step in (4500,4950,5400):
    proof=f.observe_boundary(r,p,q['report'],g,w,step,gen,lambda *x:1.)
    self.assertEqual(proof['actual_updates'],450);self.assertFalse(proof['safety_failed']);self.assertEqual(proof['pending_count'],23)
   self.assertEqual(len(q['report']['goal_sequence']['stage_observations']),12)
   self.assertEqual(q['report']['goal_sequence']['new_steps_executed'],1350)
   with self.assertRaises(ValueError):f.observe_boundary(r,p,q['report'],g,w,5850,gen,lambda *x:1.)
 def test_six_and_nine_history_digest_tampering_rejected(self):
  temp,r,p,g,c,n,b,records,w,gen=fixture4050()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=f.derive_payload(p,c,n,b,records)
   for edit in (lambda x:x['report']['goal_sequence_sources']['lineage_parent'].update(parent_history_sha256='wrong'),
      lambda x:x['report']['goal_sequence_sources']['lineage_parent']['ancestor_lineage'].update(parent_stage_sha256='wrong'),
      lambda x:x['report']['goal_sequence']['stage_observations'][0].update(pending_count=0),
      lambda x:x['optimizer']['state'][0]['step'].fill_(2700)):
    bad=copy.deepcopy(q);edit(bad)
    with self.assertRaises(ValueError):f.validate_resume_history(bad,bad['report'],g)
 def test_ordinary_product5400_is_rejected_before_runtime_ready(self):
  with self.assertRaises(ValueError):evolution._consolidate_loaded(None,steps=5400,goal_sequence=None)
 def test_5400_with_wrong_existing_protocol_cannot_bypass_explicit_budget(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'bad.json';p.write_text(json.dumps({'protocol':'unknown'}))
   with self.assertRaises(ValueError):evolution._consolidate_loaded(None,steps=5400,goal_sequence=str(p))

if __name__=='__main__':unittest.main()
