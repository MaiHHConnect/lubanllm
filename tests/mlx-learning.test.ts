import {test,expect} from 'bun:test';
import {mkdtempSync,mkdirSync,readFileSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {MLXMaterialShortage,mlxLearningRecordKey,selectMLXLearningRecords,MLX_LEARNING_PROTOCOL,mlxRead,assertMLXTrainingQualification,MLX_TRAINING_GATES,type MLXSource} from '../backend/mlx-learning-source.ts';
import {decideMLXQuality,MLX_QUALITY_GATES,mlxCandidateAction} from '../backend/mlx-learning-quality.ts';
import {MLXLearningJournal,mlxRuns,mlxPending,recoverMLXCommitPointers} from '../backend/mlx-learning-storage.ts';
import {Memory,ledgerParse,ledgerStringify} from '../backend/memory.ts';
import {MLX_LEARNING_POLICY,MLX_NET_GATES,mlxNetProgress,mlxKnowledgeSample,mlxNextKnowledge} from '../backend/mlx-learning-policy.ts';
import {privateJSON} from '../backend/settings.ts';
import {mlxGenerationComplete,MLXLearningClient} from '../backend/mlx-learning-client.ts';
import {openMLXLearningRuntime} from '../backend/mlx-learning-runtime.ts';

const hash='a'.repeat(64),row=(id:string,purpose='training')=>({id,prompt:'question '+id,answer:'answer '+id,source:'human',purpose,approved:true,training_allowed:true,provenance:JSON.stringify({verified_by:"local requester",base_identity:hash,generation_check:{type:'exact',expected:'answer '+id,max_new_tokens:16}})});
const records=()=>({train:Array.from({length:16},(_,i)=>row('t'+i)),valid:[row('v0','validation'),row('v1','validation')],protection:[row('old','validation')]});
const source=(root='/tmp',store='/tmp/store')=>({identity:hash,root,store,core:{version:'core-123456789abc',identity:hash,sha256:hash},retention:{groups:{new_validation:[],replay_validation:[row('old','validation')]},regression_validation:[]},active:{id:'trial-mlx-123456789abc',identity:hash,core_version:'core-123456789abc',width:64},expert:{width:64,sha256:hash},routerPointer:{id:'router-123456789abc',identity:hash,sha256:hash},router:{id:'router-123456789abc',sha256:hash,calibrated:true,feature_source:'mlx-q4-prefix'},proof:{schema:'mlx-expert-local-trial-v1'},bound:{identity:hash,core_version:'core-123456789abc',core_file_sha256:hash,expert_id:'trial-mlx-123456789abc',expert_file_sha256:hash,router_id:'router-123456789abc',router_sha256:hash,mlx_model_sha256:hash,worker_binary_sha256:hash},assertUnchanged(){}}as unknown as MLXSource);
const observations=(rows:any[])=>rows.map((r,i)=>({id:r.id,answer:r.answer,passed:true,complete:true,reference_used:false,tokens:[i,248044],input_tokens:[i,1]}));
test('MLX learning refuses an old, compiled or incorrectly selected observation worker',async()=>{
 for(const selected of [{event:'train_selected',source:'core'},
  {event:'train_selected',source:'core',compilation_enabled:true},
  {event:'train_selected',source:'parent',compilation_enabled:false}]){
  const client=new MLXLearningClient(source(),'/model',hash),calls:string[]=[];
  client.command=async(op)=>{calls.push(op);return selected;};
  await expect(client.generate(row('probe'),'core')).rejects.toThrow('非编译');
  expect(calls).toEqual(['train_select']);
 }
});
test('eager MLX observations still generate independently twice with original input and budget',async()=>{
 const client=new MLXLearningClient(source(),'/model',hash),calls:{op:string;body:any}[]=[];
 client.command=async(op,body={})=>{
  calls.push({op,body});
  if(op==='train_select')return{event:'train_selected',source:body.source,compilation_enabled:false};
  if(op==='prepare')return{event:'prepared',tokens:[1,2,3]};
  if(op==='generate')return{event:'result',tokens:[100,248046],finish_reason:'stop'};
  if(op==='decode')return{event:'decoded',text:'answer probe'};
  throw Error('unexpected command');
 };
 const a=await client.generate(row('probe'),'core'),b=await client.generate(row('probe'),'core');
 expect(a).toEqual(b);expect(a.passed).toBe(true);
 const generations=calls.filter(value=>value.op==='generate');expect(generations).toHaveLength(2);
 for(const {body}of generations)expect(body).toEqual({tokens:[1,2,3],max_tokens:16,eos_ids:[248044,248046],temperature:0,seed:42,cache_reuse:false});
 expect(calls.filter(value=>value.op==='prepare')[0]!.body).toEqual({messages:[{role:'user',content:'question probe'}]});
});
test('MLX opening borrows the RPC ledger without blocking its owner or closing it after source failure',async()=>{
 const root=mkdtempSync(join(tmpdir(),'mlx-rpc-open-')),memory=new Memory(root);memory.model_identity='previous';
 try{const transaction=(await memory.dispatch('db_begin',[],{},hash)).transaction;await memory.dispatch('db_execute',[],{sql:"INSERT INTO events VALUES ('owner','fixture','{}',1)"},hash,transaction);
  let settled=false;const opening=openMLXLearningRuntime({root,home:join(root,'home'),store:join(root,'store'),model:join(root,'model'),binary:join(root,'worker'),identity:hash,work:join(root,'work'),memory,routes:async()=>{throw Error('must not reach model execution');}}).then(()=>{throw Error('expected invalid source');},error=>{settled=true;return error;});
  await new Promise<void>((done,fail)=>setTimeout(()=>{expect(settled).toBe(false);memory.dispatch('db_close',[],{},hash,transaction).then(()=>done(),fail);},10));
  const error=await opening;expect(error.name).toBe('AssertionError');expect(error.stack).toContain('source (');expect(memory.model_identity).toBe('previous');
  expect(await memory.dispatch('events')).toEqual([]);await memory.dispatch('event',['still-open',{}]);expect(memory.events()[0]!.kind).toBe('still-open');
 }finally{memory.close();rmSync(root,{recursive:true,force:true});}
});
test('MLX accepts both actual Qwen chat termination tokens but never length or cancellation',()=>{for(const eos of [248044,248046])expect(mlxGenerationComplete({finish_reason:'stop',tokens:[100,eos]})).toBe(true);for(const result of [{finish_reason:'length',tokens:[248046]},{finish_reason:'cancelled',tokens:[248044]},{finish_reason:'stop',tokens:[]},{finish_reason:'stop',tokens:[100]}])expect(mlxGenerationComplete(result)).toBe(false);});
test('restored step64 validates the saved candidate without another Adam, including cancelled validation',()=>{
 const drive=(saved:number,cancelValidation=false)=>{let step=saved,validated=0,updates=0,checks=0;const candidates:number[]=[];for(;;){const action=mlxCandidateAction(step,validated);if(action==='done')return{step,updates,checks,candidates};if(action==='train'){step++;updates++;continue;}checks++;if(cancelValidation)return{step,updates,checks,candidates};validated=step;if(step===64)candidates.push(step);}};
 expect(drive(64)).toEqual({step:64,updates:0,checks:1,candidates:[64]});const paused=drive(64,true);expect(paused).toEqual({step:64,updates:0,checks:1,candidates:[]});expect(drive(paused.step)).toEqual({step:64,updates:0,checks:1,candidates:[64]});expect(drive(63)).toEqual({step:64,updates:1,checks:1,candidates:[64]});expect(mlxCandidateAction(8,0)).toBe('validate');
});
test('an errored MLX checkpoint remains pending and cannot be silently replaced by a fresh round',()=>{const root=mkdtempSync(join(tmpdir(),'luban-mlx-resume-')),memory=new Memory(root),s=source(root,join(root,'store'));memory.model_identity=hash;try{const j=new MLXLearningJournal(memory,s,records());j.update('running',{checkpoint:{path:'/saved/step64',step:64,candidate_index:0}});j.update('error',{owner:null,error:'validation interrupted'});const pending=mlxPending(memory,hash)!;expect(pending.id).toBe(j.id);expect(()=>new MLXLearningJournal(memory,s,records())).toThrow('必须恢复');const resumed=new MLXLearningJournal(memory,s,records(),pending);resumed.update('running');expect(resumed.id).toBe(j.id);expect(resumed.body.checkpoint.step).toBe(64);}finally{memory.close();rmSync(root,{recursive:true,force:true});}});
test('native MLX training qualification binds real current executable/core/model and every safety result',()=>{const binding={worker_binary_sha256:hash,core_file_sha256:hash,mlx_model_sha256:hash},proof={schema:'mlx-native-training-qualification-v1',complete:true,...binding,gates:Object.fromEntries(MLX_TRAINING_GATES.map(key=>[key,true]))};expect(()=>assertMLXTrainingQualification(proof,binding)).not.toThrow();for(const key of MLX_TRAINING_GATES)expect(()=>assertMLXTrainingQualification({...proof,gates:{...proof.gates,[key]:false}},binding)).toThrow('资格');expect(()=>assertMLXTrainingQualification({...proof,worker_binary_sha256:'b'.repeat(64)},binding)).toThrow('资格');});
test('MLX quality cannot substitute scalar loss or old precision receipts for complete generation',()=>{
 const r=records(),all=[...r.train,...r.valid,...r.protection],pass=observations(all),routes=new Map(all.map(r=>[r.id,'neural_expert']));const input={records:r,baseline:[pass,pass],parent:[[],[]],candidate:[pass,pass],baselineLoss:1,candidateLoss:.9,replayBefore:1,replayAfter:1,routes};expect(decideMLXQuality(input).accepted).toBe(true);
 for(const edit of [()=>{const bad=structuredClone(pass);bad[18]!.passed=false;return{candidate:[bad,bad]};},()=>{const bad=structuredClone(pass);bad[0]!.tokens=[55,248044];return{candidate:[pass,bad]};},()=>({candidateLoss:.999}),()=>({replayAfter:1.2}),()=>{const bad=structuredClone(pass);bad[16]!.passed=false;return{baseline:[bad,bad],routes:new Map([...routes].map(([k,v])=>[k,k==='v0'?'local_core':v]))};}])expect(decideMLXQuality({...input,...edit()}).accepted).toBe(false);
 expect(decideMLXQuality({...input,routes:new Map([...routes].map(([k,v])=>[k,k==='v0'?'local_core':v]))}).accepted).toBe(true);
});
test('MLX allows one ordinary soft protection regression when two new items are mastered',()=>{
 const r=records();r.protection=Array.from({length:5},(_,i)=>row('soft-'+i,'validation'));
 const all=[...r.train,...r.valid,...r.protection],pass=observations(all),baseline=[structuredClone(pass),structuredClone(pass)],candidate=[structuredClone(pass),structuredClone(pass)];
 for(const id of ['t0','t1']) for(const part of [baseline[0],baseline[1]]) {const proof=part!.find(v=>v.id===id)!;proof.answer='old';proof.passed=false;}
 for(const part of [candidate[0],candidate[1]]) {const proof=part!.find(v=>v.id==='soft-4')!;proof.answer='old';proof.passed=false;}
 const routes=new Map(all.map(value=>[value.id,'neural_expert']));
 const result=decideMLXQuality({records:r,baseline,parent:[[],[]],candidate,baselineLoss:1,candidateLoss:.9,replayBefore:1,replayAfter:1,routes});
 expect(result.accepted).toBe(true); expect(result.gates.mlx_protection_generation).toBe(true);
});
test('first daily learning has no invented prior mastery and rejects unexpected replay',()=>{
 const r=records(),s=source(),rows=[...r.train,...r.valid];expect(selectMLXLearningRecords(rows,s).train.length).toBe(16);expect(selectMLXLearningRecords(rows,s).protection).toEqual([]);
 expect(()=>selectMLXLearningRecords(rows,s,[],r)).toThrow('protection');const conflict=structuredClone(rows);conflict[16]!.prompt=conflict[0]!.prompt;expect(()=>selectMLXLearningRecords(conflict,s)).toThrow('leakage');
});
test('MLX durable publication restores its three pointers and refuses to overwrite a rival',()=>{
 const root=mkdtempSync(join(tmpdir(),'luban-mlx-learning-')),store=join(root,'modules/model');mkdirSync(store,{recursive:true});const s=source(root,store),memory=new Memory(root);memory.model_identity=hash;
 try{privateJSON(join(store,'active-core.json'),s.core);privateJSON(join(store,'active.json'),s.active);privateJSON(join(store,'active-router.json'),s.routerPointer);privateJSON(join(root,'runtime/mlx/expert-proof.json'),s.proof);writeFileSync(join(store,s.routerPointer.id+'.safetensors'),'router');const weights=join(root,'weights');writeFileSync(weights,'candidate');
  const journal=new MLXLearningJournal(memory,s,records());journal.update('running');expect(journal.body.arguments.protocol).toBe(MLX_LEARNING_PROTOCOL);
  expect(()=>journal.prepare({learning_policy:MLX_LEARNING_POLICY,promoted:true,quality_checked:true,gates:{}},weights)).toThrow('quality');
  const publication=journal.prepare({learning_policy:MLX_LEARNING_POLICY,promoted:true,quality_checked:true,gates:Object.fromEntries(MLX_NET_GATES.map(k=>[k,true])),chosen_width:64,candidates:[]},weights);expect(publication.router.feature_source).toBe('mlx-q4-prefix');journal.apply();expect(mlxRead(join(store,publication.router.id+'.json')).feature_source).toBe('mlx-q4-prefix');expect(mlxRead(join(store,'active.json')).id).toBe('learned-'+journal.id);
  recoverMLXCommitPointers(memory,{root,store,identity:hash});expect(mlxRead(join(store,'active.json'))).toEqual(s.active);expect(mlxRead(join(root,'runtime/mlx/expert-proof.json'))).toEqual(s.proof);
  journal.apply();privateJSON(join(store,'active.json'),{id:'rival'});expect(()=>journal.rollback('failure')).toThrow('rival');expect(mlxRead(join(store,'active.json')).id).toBe('rival');expect(mlxRuns(memory,hash)[0]!.state).toBe('committing');
 }finally{memory.close();rmSync(root,{recursive:true,force:true});}
});

test('daily MLX material excludes experiments and reports actual train and independent validation shortage',()=>{
 const r=records(), historical=[...r.train,...r.valid].map(row=>({...row,provenance:JSON.stringify({...JSON.parse(row.provenance),dataset:'old-study'})}));
 expect(()=>selectMLXLearningRecords([...historical,...r.train.slice(0,3)],source())).toThrow('训练 3 / 16 条，独立验证 0 / 2 条');
 try{selectMLXLearningRecords([...historical,...r.train.slice(0,3)],source());throw new Error('expected shortage');}catch(error){expect(error).toBeInstanceOf(MLXMaterialShortage);expect((error as MLXMaterialShortage).training).toBe(3);expect((error as MLXMaterialShortage).validation).toBe(0);}
 const selected=selectMLXLearningRecords([...historical,...r.train,...r.valid],source());
 expect(selected.train.map(row=>row.id)).toEqual(r.train.map(row=>row.id));
 expect(selected.valid).toEqual(r.valid);
});

test('real MLX journals retain automatic cohort identity through rejection and reopening',()=>{
 const root=mkdtempSync(join(tmpdir(),'mlx-attempt-')),memory=new Memory(root),s=source(root,join(root,'store'));memory.model_identity=hash;
 try{const j=new MLXLearningJournal(memory,s,records(),undefined,{signature:hash,scheduled_occurrence:null});j.update('rejected',{owner:null});const persisted=JSON.parse(mlxRuns(memory,hash)[0]!.body);expect(persisted.automatic_signature).toBe(hash);expect(persisted.scheduled_occurrence).toBeNull();expect(persisted.kind).toBe('learning');}finally{memory.close();rmSync(root,{recursive:true,force:true});}
});

test('net learning counts held-out gains, accepts gain2 loss1 but rejects gain1 loss1 and training-only improvement',()=>{
 const r=records(),all=[...r.train,...r.valid,...r.protection],pass=observations(all);
 const baseline=[structuredClone(pass),structuredClone(pass)],candidate=[structuredClone(pass),structuredClone(pass)];
 for(const part of baseline)for(const v of part)if(['v0','v1'].includes(v.id))v.passed=false;
 for(const part of candidate)part.find(v=>v.id==='old')!.passed=false;
 const input={records:r,baseline,parent:[[],[]],candidate,baselineLoss:1,candidateLoss:.9,replayBefore:1,replayAfter:9,routes:new Map(all.map(v=>[v.id,'neural_expert'])),policy:MLX_LEARNING_POLICY};
 const result=decideMLXQuality(input);expect(result.accepted).toBe(true);expect(result.learning_progress?.net_gain).toBe(1);
 const tied=structuredClone(candidate);for(const part of tied)part.find(v=>v.id==='v1')!.passed=false;
 expect(decideMLXQuality({...input,candidate:tied}).accepted).toBe(false);
 const already=structuredClone(pass);expect(decideMLXQuality({...input,baseline:[already,already]}).accepted).toBe(false);
 // A previously mastered question lost before this round is still a debt, not erased by rebaselining.
 for(const part of baseline)part.find(v=>v.id==='old')!.passed=false;
 expect(mlxNetProgress(r,baseline,candidate).forgotten_ids).toEqual(['old']);
});
test('first round accepts genuine new validation gain without old protection or replay loss',()=>{
 const r=records();r.protection=[];const all=[...r.train,...r.valid],pass=observations(all),base=structuredClone(pass);base.find(v=>v.id==='v0')!.passed=false;
 const input={records:r,baseline:[base,base],parent:[[],[]],candidate:[pass,pass],baselineLoss:1,candidateLoss:.9,replayBefore:0,replayAfter:0,routes:new Map(all.map(v=>[v.id,'neural_expert'])),policy:MLX_LEARNING_POLICY};
 expect(decideMLXQuality(input).accepted).toBe(true);
 expect(decideMLXQuality({...input,routes:new Map(all.map(v=>[v.id,'local_core']))}).accepted).toBe(false);
 const broken=structuredClone(pass);broken[0]!.input_tokens=[999];expect(decideMLXQuality({...input,candidate:[broken,broken]}).accepted).toBe(false);
});
test('adopted knowledge rotates through checks and forgotten questions requeue their original authorized training',()=>{
 const r=records(),pool=Array.from({length:20},(_,i)=>({...row('p'+String(i).padStart(2,'0'),'validation'),provenance:JSON.stringify({...JSON.parse(row('x').provenance),learning_train_ids:r.train.map(v=>v.id)})}));
 const previous:any={learning_policy:MLX_LEARNING_POLICY,knowledge_pool:pool,review_cursor:16,relearning_records:[],covered_keys:[]};
 expect(mlxKnowledgeSample(previous).map(v=>v.id)).toEqual([...pool.slice(16),...pool.slice(0,12)].map(v=>v.id));
 r.protection=[pool[0]!];const pass=observations([...r.train,...r.valid,...r.protection]);pass.find(v=>v.id==='p00')!.passed=false;
 const next=mlxNextKnowledge(previous,r,[pass,pass]);expect(next.knowledge_pool.some(v=>v.id==='p00')).toBe(false);expect(next.relearning_records.map(v=>v.id)).toEqual(['p00']);
 expect(next.knowledge_pool.find(v=>v.id==='v0')?.provenance).toContain('learning_train_ids');
 const s=source();s.acceptedReport={...previous,...next,covered_keys:r.train.map(v=>mlxLearningRecordKey(v))};
 const selected=selectMLXLearningRecords([...r.train,...r.valid,pool[0]!],s);
 expect(selected.train.map(v=>v.id)).toEqual(r.train.map(v=>v.id));expect(selected.valid[0]!.id).toBe('p00');
 expect(selected.train.every(v=>v.purpose==='training')).toBe(true);
 const repaired=mlxNextKnowledge(s.acceptedReport,{train:r.train,valid:[pool[0]!],protection:[]},[observations([pool[0]!]),observations([pool[0]!])]);expect(repaired.relearning_records).toEqual([]);
});
test('TRAIN can use core at inference while held-out gains still require reachable experts',()=>{
 const r=records();r.protection=[];const all=[...r.train,...r.valid],pass=observations(all),base=structuredClone(pass),after=structuredClone(pass);
 for(const value of base)if(['t0','v0','v1'].includes(value.id))value.passed=false;
 after[0]=structuredClone(base[0]!);
 const routes=new Map(all.map(value=>[value.id,value.id==='t0'?'local_core':'neural_expert']));
 const input={records:r,baseline:[base,base],parent:[[],[]],candidate:[after,after],baselineLoss:1,candidateLoss:.9,replayBefore:0,replayAfter:0,routes,policy:MLX_LEARNING_POLICY};
 expect(decideMLXQuality(input).accepted).toBe(true);
 const blocked=new Map(routes);blocked.set('v0','local_core');
 expect(decideMLXQuality({...input,routes:blocked}).gates.mlx_routes_available).toBe(false);
});

test('resumed MLX quality accepts ledger floats without accepting coercible invalid losses',()=>{
 const r=records(),all=[...r.train,...r.valid,...r.protection],pass=observations(all),base=structuredClone(pass);base.find(v=>v.id==='v0')!.passed=false;
 const fresh={records:r,baseline:[base,base],parent:[[],[]],candidate:[pass,pass],baselineLoss:.2,candidateLoss:.1,replayBefore:.05,replayAfter:.05,routes:new Map(all.map(v=>[v.id,'neural_expert'])),policy:MLX_LEARNING_POLICY};
 const {routes,...persisted}=fresh,resumed={...ledgerParse(ledgerStringify(persisted)),routes};
 expect(resumed.baselineLoss).toBeInstanceOf(Number);expect(decideMLXQuality(resumed)).toEqual(decideMLXQuality(fresh));
 for(const policy of [MLX_LEARNING_POLICY,undefined]){
  expect(decideMLXQuality({...resumed,policy})).toEqual(decideMLXQuality({...fresh,policy}));
  for(const bad of ['0.2',null,true,NaN,Infinity,-.1,new Number(NaN)])expect(()=>decideMLXQuality({...resumed,policy,baselineLoss:bad})).toThrow('非负有限数');
 }
});
