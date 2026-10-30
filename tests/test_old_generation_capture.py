import copy,json,threading,unittest
from types import SimpleNamespace
import torch
from luban.evolution import _capture_replay_subspace
from luban.goal_fresh import validated_protection_tokens as _validated_protection_tokens,content_sha as _sha,tensor_sha as _tensor_sha
from luban.evolution import _project_core_delta,_constraint_residual
class Tokenizer:
 def __len__(self):return 5
 def decode(self,raw,**kwargs):return 'OK' if raw==[1,3] else 'BAD'
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__(); self.down=torch.nn.Linear(4,2,bias=False)
  self.embedding=torch.nn.Embedding.from_pretrained(torch.eye(4).tolist() and torch.eye(4),freeze=True)
  self.model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=self.down))])
  self.generation_config=SimpleNamespace(eos_token_id=[3]);self.calls=[];self.eval()
 def forward(self,ids,**kwargs):
  self.calls.append(ids.tolist());return self.down(self.embedding(ids))
def fixture():
 m=Model();r=SimpleNamespace(model=m,branch=SimpleNamespace(expert=None),tokenizer=Tokenizer(),stop=threading.Event(),job={},training_session=None,tokenizer_sha256='tokenizer')
 r.original_tensor_sha256=_tensor_sha(m.down.weight)
 r._tokens=lambda *args,**kwargs:torch.tensor([[0]])
 r._sample_tokens=lambda s:(torch.tensor([[0,3]]),torch.tensor([[-100,3]]))
 s={'id':'known-old','prompt':'old','answer':'human gold is not fed','purpose':'validation','training_allowed':False,'approved':True,'source':'human','provenance':json.dumps({'rights_basis':'project-authored/explicit local old-state protection','generation_check':{'type':'exact','expected':'OK','max_new_tokens':2}})}
 row={'id':'known-old','answer':'OK','passed':True,'complete':True,'raw_completion_token_ids':[1,3],'completion_tokens':2,'max_new_tokens':2,'decoded_completion_unstripped':'OK','branch_enabled':False,'reference_used':False,'thinking':'off','do_sample':False}
 d={'policy':'known-old-generated-states-v1','scope':'known_old_protection_only','original_tensor_sha256':r.original_tensor_sha256,'baseline_tensor_sha256':r.original_tensor_sha256,'tokenizer_sha256':'tokenizer','records':[s],'results':[row],'authorized_record_sha256':{'known-old':_sha(s)}}
 train={'id':'train','purpose':'training','provenance':json.dumps({'evolution_role':'replay'})}
 return r,d,train
class CaptureTests(unittest.TestCase):
 def test_actual_path_changes_protected_directions_without_ce(self):
  r,d,t=fixture();old,a=_capture_replay_subspace(r,r.model.down.weight,[t]);new,b=_capture_replay_subspace(r,r.model.down.weight,[t],protection_capture=d)
  self.assertEqual((a['rank'],b['rank']),(1,2));self.assertEqual(b['training_ids'],['train']);self.assertEqual(b['protection_ids'],['known-old']);self.assertEqual(b['positions_per_sample'],[1]);self.assertEqual(b['protection_positions_per_sample'],[2]);self.assertFalse(d['records'][0]['training_allowed'])
  reference=torch.zeros(2,4);delta=torch.zeros(2,4);delta[:,1]=1
  master=delta.clone();_project_core_delta(master,reference,old);self.assertEqual(float(master[:,1].abs().sum()),2)
  _project_core_delta(master,reference,new);self.assertLess(float(master[:,1].abs().sum()),1e-5)
  self.assertFalse(any(4 in row[0] for row in r.model.calls));self.assertFalse(r.model.training)
 def test_large_added_directions_do_not_delete_small_original_train_span(self):
  r,d,t=fixture()
  with torch.no_grad():r.model.embedding.weight[1].mul_(1e8)
  basis,report=_capture_replay_subspace(r,r.model.down.weight,[t],protection_capture=d)
  self.assertEqual(report['training_rank'],1);self.assertEqual(report['rank'],2)
  self.assertLess(report['old_span_max_abs'],2e-5)
 def test_validation_cannot_be_smuggled_into_ordinary_train(self):
  r,d,t=fixture()
  with self.assertRaises(ValueError):_capture_replay_subspace(r,r.model.down.weight,d['records'])
  self.assertEqual(len(r.model.down._forward_pre_hooks),0)
 def test_missing_terminal_eos_rejected(self):
  r,d,t=fixture();d['results'][0]['raw_completion_token_ids']=[1,1]
  with self.assertRaises(ValueError):_validated_protection_tokens(r,d)
 def test_failed_or_branch_output_rejected(self):
  for field,value in [('passed',False),('complete',False),('branch_enabled',True),('reference_used',True),('decoded_completion_unstripped','forged')]:
   r,d,t=fixture();d['results'][0][field]=value
   with self.assertRaises(ValueError):_validated_protection_tokens(r,d)
 def test_new_validation_unallowlisted_rejected_before_execution(self):
  r,d,t=fixture();d['records'][0]['id']='new-validation';d['results'][0]['id']='new-validation'
  with self.assertRaises(ValueError):_capture_replay_subspace(r,r.model.down.weight,[t],protection_capture=d)
  self.assertFalse(r.model.calls);self.assertFalse(r.model.down._forward_pre_hooks)
 def test_changed_record_and_reference_rejected(self):
  for edit in [lambda d:d['records'][0].update(prompt='changed'),lambda d:d.update(baseline_tensor_sha256='wrong'),lambda d:d.update(tokenizer_sha256='wrong')]:
   r,d,t=fixture();edit(d)
   with self.assertRaises(ValueError):_validated_protection_tokens(r,d)
 def test_non_original_weight_rejected(self):
  r,d,t=fixture()
  with torch.no_grad():r.model.down.weight.add_(1)
  with self.assertRaises(ValueError):_capture_replay_subspace(r,r.model.down.weight,[t],protection_capture=d)
 def test_quantization_residual_is_not_zero_guarantee(self):
  torch.manual_seed(7);ref=torch.randn(2,4).bfloat16().float();basis=torch.linalg.qr(torch.randn(4,2)).Q;master=ref+torch.randn(2,4)*.05
  _project_core_delta(master,ref,basis);candidate=master.bfloat16();res=_constraint_residual(candidate,master,ref,basis)
  self.assertLess(res['master_frobenius'],1e-6);self.assertGreater(res['quantized_frobenius'],1e-5)
if __name__=='__main__':unittest.main()
