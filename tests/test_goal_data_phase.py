import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from luban import goal_fresh as f,evolution
from test_fresh_continuation_v4 import fixture5400

# Tiny CPU controller state. No actual6750 model result or model generation here.
def fixture_phase():
 temp,r,p,g,c,n,b,records,w,gen=fixture5400()
 with patch.object(f.common,'SHAPE',(2,4)):q=f.derive_payload(p,c,n,b,records)
 for step in (5850,6300):f.observe_boundary(r,p,q['report'],g,w,step,gen,lambda *x:1.)
 def all24(runtime,rows):
  out=gen(runtime,rows)
  for row in out:row['passed']=True
  return out
 f.observe_boundary(r,p,q['report'],g,w,6750,all24,lambda *x:1.)
 master=torch.nn.Parameter(q['master'].clone());optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6);optimizer.load_state_dict(q['optimizer'])
 for _ in range(1350):optimizer.zero_grad();master.square().sum().backward();optimizer.step()
 q.update(step=6750,master=master.detach().clone(),optimizer=optimizer.state_dict(),new_exposures={v['id']:3375//24+int(i<3375%24) for i,v in enumerate(g['new_training'])})
 for key,rows in g.items():
  for row in rows:row.update(prompt=row['id'],answer='OK',purpose='validation' if key.endswith('validation') else 'training',provenance=json.dumps({'generation_check':{'type':'exact','expected':'OK'}}))
 for row in records['protection']:row.update(prompt=row['id'],answer='OK',purpose='validation',provenance=json.dumps({'generation_check':{'type':'exact','expected':'OK'}}))
 cap={'records':records['protection']};cap_path=Path(q['report']['goal_sequence_sources']['protection_capture']['path']);cap_path.write_text(json.dumps(cap));q['report']['replay_subspace']['protection_capture_sha256']=f.content_sha(cap)
 b=dict(b,records_sha256=f.content_sha(records));closed={'binding':b,'payload':q,'rng':c['rng'],'mps_rng':c['mps_rng']}
 record_path=Path(temp.name).resolve()/'records.json';record_path.write_text(json.dumps(records))
 ng=copy.deepcopy(g)
 ng['new_training'] += [{'id':f'phase_new{i}','prompt':f'phase_new{i}','answer':'OK','purpose':'training','provenance':json.dumps({'generation_check':{'type':'exact','expected':'OK'}})} for i in range(32)]
 ng['new_validation'] += [{'id':f'phase_valid{i}','prompt':f'phase_valid{i}','answer':'OK','purpose':'validation','provenance':json.dumps({'generation_check':{'type':'exact','expected':'OK'}})} for i in range(4)]
 nr={'groups':ng,'protection':records['protection'],'retention_regressions':[{'id':f'accept{i}'} for i in range(12)]}
 files={'records':{'path':str(record_path),'sha256':f.common.file_sha(record_path)},'parent_checkpoint':{'path':'toy-only','sha256':'x'*64},'parent_result':{'path':'toy-only','sha256':'z'*64}}
 pp={**p,'protocol':f.DATA_PHASE,'new_run_id':'271828182abc','parent_run_id':q['id'],'parent_steps':6750,'total_steps':9450,'files':files,
  '_parent_report':q['report'],'_parent_binding':b,'_records':nr,'parent_binding_sha256':f.content_sha(b),'parent_report_sha256':f.content_sha(q['report']),
  'parent_history_sha256':f.content_sha(q['report']['training_selection']['results']),'parent_stage_sha256':f.content_sha(q['report']['goal_sequence']['stage_observations']),
  'parent_goal_sequence_sources_sha256':f.content_sha(q['report']['goal_sequence_sources']),'ancestor_lineage':q['report']['goal_sequence_sources']['lineage_parent'],
  'parent_new_exposures':q['new_exposures'],'original_formal_protocol':{'path':'toy-only','sha256':'a'*64},'original_formal_files':{},'material_manifest':{'path':'toy-only','sha256':'m'*64},'_files':{},
  'new_dataset_sha256':f.content_sha(ng),'new_records_sha256':f.content_sha(nr),'new_implementation_sha256':{'phase':'e'},'parent_final_tensor_sha256':f.tensor_sha(w)}
 nb={**b,'arguments':f.arguments(pp),'implementation_sha256':pp['new_implementation_sha256'],'records_sha256':pp['new_records_sha256']}
 nn={**n,'candidate':w.detach().clone(),'master':master.detach().clone(),'exp_avg':optimizer.state_dict()['state'][0]['exp_avg'].clone(),'exp_avg_sq':optimizer.state_dict()['state'][0]['exp_avg_sq'].clone(),'adam_step':torch.tensor(6750.)}
 r.training_session.body['records']=nr;calls=[]
 def generation(runtime,rows,passed=24,forgotten=False):
  assert not any(row['id'].startswith(('new_validation','phase_valid','accept')) for row in rows),'VALID/ACCEPT accessed by TRAIN observer'
  calls.extend(row['id'] for row in rows)
  output=[]
  for i,row in enumerate(rows):
   okay=(i<passed if len(rows)==56 else True) and not (forgotten and row['id']==ng['new_training'][0]['id'])
   output.append({'id':row['id'],'answer':'OK' if okay else 'wrong','passed':okay,'complete':True,'branch_enabled':False,'reference_used':False,'raw_completion_token_ids':[1]})
  return output
 return temp,r,pp,ng,closed,nn,nb,nr,w,generation,calls

def derive_with_baseline(p,c,n,b,records,r,g,w,gen):
 q=f.derive_payload(p,c,n,b,records)
 q['report']['goal_sequence']['reference_baseline_before_training']=f.observe(r,p,g,w,6750,gen,lambda *x:1.,reference=[v['id'] for v in g['new_training'][:24]],prior=[v['id'] for v in g['new_training'][:24]])
 return q

class DataPhaseTests(unittest.TestCase):
 def test_budget_has_six_phase_boundaries_not_old15(self):
  p={'protocol':f.DATA_PHASE,'_protocol_sha256':'a'*64,'parent_run_id':'03fd69e9f865'}
  self.assertEqual(f.checkpoints(p),[7200,7650,8100,8550,9000,9450]);self.assertEqual(f.arguments(p)['steps'],9450)
  meta=f.report_metadata(p);self.assertEqual(meta['phase_steps'],0);self.assertEqual(meta['cumulative_steps'],6750)
 def test_fullstate_preserved_and_new_dataset_history_separate(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   before=copy.deepcopy(c);q=f.derive_payload(p,c,n,b,records)
   for k in ('master','basis'):self.assertTrue(torch.equal(q[k],before['payload'][k]))
   for k in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(q['optimizer']['state'][0][k],before['payload']['optimizer']['state'][0][k]))
   self.assertEqual(q['step'],6750);self.assertEqual(q['report']['steps'],9450);self.assertEqual(len(q['new_exposures']),56)
   self.assertTrue(all(q['new_exposures'][k]==v for k,v in before['payload']['new_exposures'].items()));self.assertTrue(all(q['new_exposures'][v['id']]==0 for v in g['new_training'][24:]))
   self.assertFalse(q['report']['training_selection']['results']);self.assertIsNone(q['report']['training_selection']['selected_step']);self.assertFalse(q['report']['goal_sequence']['stage_observations'])
   self.assertEqual(q['report']['goal_sequence']['parent_archive']['parent_stage_sha256'],f.content_sha(before['payload']['report']['goal_sequence']['stage_observations']))
   self.assertEqual(c['payload']['report'],before['payload']['report']);self.assertEqual(c['payload']['new_exposures'],before['payload']['new_exposures']);self.assertTrue(torch.equal(c['rng'],before['rng']));self.assertTrue(torch.equal(c['mps_rng'],before['mps_rng']))
 def test_unobserved_entry_is_rejected_not_faked(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=f.derive_payload(p,c,n,b,records)
   with self.assertRaisesRegex(ValueError,'has not been observed'):f.validate_resume_history(q,q['report'],g)
 def test_true_phase_baseline_and_resume_inherit_adam6750(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=derive_with_baseline(p,c,n,b,records,r,g,w,gen);f.validate_resume_history(q,q['report'],g)
   tm=torch.nn.Parameter(torch.ones_like(q['master']));opt=torch.optim.Adam([tm],lr=1e-5,eps=1e-6)
   evolution._resume_core(SimpleNamespace(id=q['id']),q,copy.deepcopy(q['report']),w,w.detach().clone(),g,9450,tm,opt)
   self.assertEqual(float(opt.state[tm]['step']),6750);self.assertTrue(torch.equal(tm,q['master']))
   self.assertEqual(q['report']['goal_sequence']['reference_baseline_before_training']['confirmed_count'],24)
 def test_absolute6750_trace_exposure_offset_is_preserved(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp:
   source=f.phase_sources(p);base=f.phase_exposures(source,g,6750);two=f.phase_exposures(source,g,6753)
   self.assertEqual(two[g['new_training'][15]['id']],base[g['new_training'][15]['id']]+1)
   self.assertEqual(two[g['new_training'][16]['id']],base[g['new_training'][16]['id']]+1)
   self.assertEqual(sum(two.values())-sum(base.values()),2)
   final=f.phase_exposures(source,g,9450);self.assertEqual(sum(final.values())-sum(base.values()),1350)
 def test_six_complete450_pending_rounds_are_upgrades_without_valid(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=derive_with_baseline(p,c,n,b,records,r,g,w,gen)
   for step in f.checkpoints(p):
    proof=f.observe_boundary(r,p,q['report'],g,w,step,gen,lambda *x:1.)
    self.assertFalse(proof['safety_failed']);self.assertEqual(proof['actual_updates'],450);self.assertEqual(proof['round_result'],'partial')
   self.assertEqual(q['report']['goal_sequence']['phase_steps'],2700);self.assertEqual(len(q['report']['goal_sequence']['stage_observations']),6)
   f.phase_proofs(q['report'],g,records['protection'])
   self.assertFalse(any(i.startswith(('new_validation','phase_valid','accept')) for i in calls))
   with self.assertRaises(ValueError):f.observe_boundary(r,p,q['report'],g,w,9900,gen,lambda *x:1.)
 def test_prior24_forgetting_stops_next_round(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=derive_with_baseline(p,c,n,b,records,r,g,w,gen)
   proof=f.observe_boundary(r,p,q['report'],g,w,7200,lambda runtime,rows:gen(runtime,rows,forgotten=True),lambda *x:1.)
   self.assertTrue(proof['safety_failed']);self.assertEqual(proof['round_result'],'protection_failed')
   f.phase_proofs(q['report'],g,records['protection'])
   with self.assertRaises(ValueError):f.observe_boundary(r,p,q['report'],g,w,7650,gen,lambda *x:1.)
 def test_first_all56_selects_and_does_not_continue(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=derive_with_baseline(p,c,n,b,records,r,g,w,gen)
   proof=f.observe_boundary(r,p,q['report'],g,w,7200,lambda runtime,rows:gen(runtime,rows,passed=56),lambda *x:1.)
   self.assertEqual(proof['confirmed_count'],56);self.assertEqual(q['report']['training_selection']['selected_step'],7200)
   f.phase_proofs(q['report'],g,records['protection'])
   with self.assertRaises(ValueError):f.observe_boundary(r,p,q['report'],g,w,7650,gen,lambda *x:1.)
 def test_archive_judge_exposure_or_adam_tamper_rejected(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp,patch.object(f.common,'SHAPE',(2,4)):
   q=derive_with_baseline(p,c,n,b,records,r,g,w,gen)
   edits=[lambda x:x['report']['goal_sequence']['parent_archive'].update(parent_stage_sha256='wrong'),lambda x:x['new_exposures'].update(phase_new0=1),lambda x:x['optimizer']['state'][0]['step'].fill_(0),lambda x:x['report']['goal_sequence']['reference_baseline_before_training']['observations'][0]['new'][0].update(answer='wrong')]
   for edit in edits:
    bad=copy.deepcopy(q);edit(bad)
    with self.assertRaises(ValueError):f.validate_resume_history(bad,q['report'],g)
 def test_validation_reference_uses6750_not_kl_B45(self):
  temp,r,p,g,c,n,b,records,w,gen,calls=fixture_phase()
  with temp:
   changed=copy.deepcopy(c);changed['payload']['master'].fill_(1);p['_parent_report']=changed['payload']['report'];p['parent_final_tensor_sha256']=f.tensor_sha(torch.ones_like(w))
   before=w.detach().clone()
   with patch.object(f.torch,'load',return_value=changed):value=f.validation_reference(p,w.detach())
   self.assertTrue(torch.equal(value,torch.ones_like(w)));self.assertTrue(torch.equal(w,before))
 def test_missing_closure_wrong_budget_and_ordinary9450_are_rejected(self):
  with self.assertRaises(ValueError):evolution._consolidate_loaded(None,steps=9450)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d).resolve()/'protocol.json';base={'protocol':f.DATA_PHASE,'status':'FROZEN_NEW_DATA_PHASE','parent_steps':6750,'total_steps':9450,'segment_steps':450,'policy_version':f.POLICY,'parent_run_id':'03fd69e9f865','new_run_id':'271828182abc','files':{},'new_files':{}}
   for edit in ({},{'total_steps':9900},{'parent_steps':True},{'parent_run_id':'1d80fd7152eb'}):
    p.write_text(json.dumps({**base,**edit}))
    with self.assertRaises(ValueError):f.load_verified(p)
   p.write_text(json.dumps({'protocol':'unknown'}))
   with self.assertRaises(ValueError):evolution._consolidate_loaded(None,steps=9450,goal_sequence=str(p))

if __name__=='__main__':unittest.main()
