import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from luban import goal_fresh as f,evolution
from test_fresh_continuation_v3 import fixture4050

# Controller toy5400, explicitly not an actual model result.
def fixture5400():
 temp,r,p,g,c,n,b,records,w,gen=fixture4050()
 with patch.object(f.common,'SHAPE',(2,4)):q=f.derive_payload(p,c,n,b,records)
 for step in (4500,4950,5400):f.observe_boundary(r,p,q['report'],g,w,step,gen,lambda *x:1.)
 master=torch.nn.Parameter(q['master'].clone());optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6);optimizer.load_state_dict(q['optimizer'])
 for _ in range(1350):optimizer.zero_grad();master.square().sum().backward();optimizer.step()
 q.update(step=5400,master=master.detach().clone(),optimizer=optimizer.state_dict(),new_exposures={v['id']:2700//24+int(i<2700%24) for i,v in enumerate(g['new_training'])})
 c2={'binding':b,'payload':q,'rng':c['rng'].clone(),'mps_rng':c['mps_rng'].clone()}
 p4={**p,'protocol':f.RESUME_PROTOCOL_V4,'parent_run_id':q['id'],'new_run_id':'141421356abc','parent_steps':5400,'total_steps':6750,
     '_parent_report':q['report'],'_parent_binding':b,'parent_binding_sha256':f.content_sha(b),'parent_report_sha256':f.content_sha(q['report']),
     'parent_history_sha256':f.content_sha(q['report']['training_selection']['results']),'parent_implementation_sha256':b['implementation_sha256'],
     'new_implementation_sha256':{'v4':'d'},'parent_goal_sequence_sources_sha256':f.content_sha(q['report']['goal_sequence_sources']),
     'ancestor_lineage':copy.deepcopy(q['report']['goal_sequence_sources']['lineage_parent'])}
 b4={**b,'arguments':f.arguments(p4),'implementation_sha256':p4['new_implementation_sha256']}
 n4={**n,'master':master.detach().clone(),'exp_avg':optimizer.state_dict()['state'][0]['exp_avg'].clone(),'exp_avg_sq':optimizer.state_dict()['state'][0]['exp_avg_sq'].clone(),'adam_step':torch.tensor(5400.)}
 return temp,r,p4,g,c2,n4,b4,records,w,gen

class FreshV4Tests(unittest.TestCase):
 def test_actual_v3_wrapper_remains_readonly_compatible(self):
  p=Path('/Volumes/移动数据库/lubanllm/experiments/goal-prospective-fixed450-v3-protocol-20261011T080604Z-c6fcbf/protocol.json')
  value=f.load_verified(p)
  self.assertEqual(value['new_run_id'],'1d80fd7152eb');self.assertEqual(f.metadata(value)['parent_steps'],4050)
  self.assertEqual(f.source_metadata(value)['scope'],'closed fresh4050 full-state continuation; originalB45 reference/basis; no failed old v3 ancestry')
 def test_toy5400_preserves_complete_state_twelve_rounds_and_nested_ancestry(self):
  temp,r,p,g,c,n,b,records,w,gen=fixture5400()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   old=copy.deepcopy(c);q=f.derive_payload(p,c,n,b,records)
   self.assertEqual(q['step'],5400);self.assertEqual(q['report']['steps'],6750)
   for k in ('master','basis'):self.assertTrue(torch.equal(q[k],old['payload'][k]))
   for k in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(q['optimizer']['state'][0][k],old['payload']['optimizer']['state'][0][k]))
   self.assertEqual(q['new_exposures'],old['payload']['new_exposures'])
   self.assertEqual(q['report']['goal_sequence']['stage_observations'],old['payload']['report']['goal_sequence']['stage_observations'])
   self.assertEqual(q['report']['training_selection']['results'],old['payload']['report']['training_selection']['results'])
   self.assertTrue(torch.equal(c['rng'],old['rng']));self.assertTrue(torch.equal(c['mps_rng'],old['mps_rng']))
   lineage=q['report']['goal_sequence_sources']['lineage_parent']
   self.assertEqual(lineage['ancestor_lineage'],p['ancestor_lineage'])
   self.assertEqual(lineage['parent_goal_sequence_sources_sha256'],p['parent_goal_sequence_sources_sha256'])
   f.validate_resume_history(q,q['report'],g)
 def test_complete450_boundaries_carry_pending_with_finite_budget(self):
  temp,r,p,g,c,n,b,records,w,gen=fixture5400()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=f.derive_payload(p,c,n,b,records)
   for step in (5850,6300,6750):
    proof=f.observe_boundary(r,p,q['report'],g,w,step,gen,lambda *x:1.)
    self.assertEqual(proof['actual_updates'],450);self.assertEqual(proof['pending_count'],23);self.assertFalse(proof['safety_failed'])
   self.assertEqual(len(q['report']['goal_sequence']['stage_observations']),15)
   self.assertEqual(q['report']['goal_sequence']['new_steps_executed'],1350)
   with self.assertRaises(ValueError):f.observe_boundary(r,p,q['report'],g,w,7200,gen,lambda *x:1.)
 def test_all_three_inherited_history_levels_are_immutable(self):
  temp,r,p,g,c,n,b,records,w,gen=fixture5400()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=f.derive_payload(p,c,n,b,records)
   for depth in (0,1,2):
    bad=copy.deepcopy(q);lineage=bad['report']['goal_sequence_sources']['lineage_parent']
    for _ in range(depth):lineage=lineage['ancestor_lineage']
    lineage['parent_stage_sha256']='tampered'
    with self.assertRaises(ValueError):f.validate_resume_history(bad,bad['report'],g)
 def test_unregistered6750_and_foreign_protocol_rejected(self):
  with self.assertRaises(ValueError):evolution._consolidate_loaded(None,steps=6750)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'unknown.json';p.write_text(json.dumps({'protocol':'unknown'}))
   with self.assertRaises(ValueError):evolution._consolidate_loaded(None,steps=6750,goal_sequence=str(p))
 def test_incomplete_or_foreign_parent_wrapper_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'protocol.json'
   base={'protocol':f.RESUME_PROTOCOL_V4,'status':'FROZEN_CLOSED_PARENT','parent_steps':5400,'total_steps':6750,'segment_steps':450,
    'policy_version':f.POLICY,'parent_run_id':'1d80fd7152eb','new_run_id':'141421356abc','files':{}}
   for changes in ({},{'parent_steps':4050},{'total_steps':7200},{'parent_run_id':'25bb1aa918d8'}):
    p.write_text(json.dumps({**base,**changes}))
    with self.assertRaises(ValueError):f.load_verified(p)

if __name__=='__main__':unittest.main()
