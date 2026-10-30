import {test,expect} from 'bun:test';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync,symlinkSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {readMLXLearningSource,MLX_TRAINING_GATES,isUntrainedMLXInstallation,MLX_FIRST_LEARNING_NOTICE} from '../backend/mlx-learning-source';
import {materialContentSHA} from '../backend/native-evolution-material';

test('learning readiness refuses historical BF16 routers before opening the MLX payload',()=>{
 const root=mkdtempSync(join(tmpdir(),'mlx-feature-source-')),store=join(root,'store'),modelPath=join(root,'model'),binary=join(root,'worker'),identity='a'.repeat(64),sha='b'.repeat(64);
 const core='core-123456789abc',expert='learned-123456789abc',router='router-123456789abc';
 const write=(path:string,value:unknown)=>{mkdirSync(join(path,'..'),{recursive:true});writeFileSync(path,JSON.stringify(value));};
 try{
  write(join(store,'active-core.json'),{version:core,identity,sha256:sha});
  write(join(store,core+'.json'),{version:core,identity,sha256:sha,report:'legacy-report.json'});
  write(join(store,'legacy-report.json'),{});
  const retained={schema_version:1,identity,version:core,groups:{new_training:[],new_validation:[],replay_training:[],replay_validation:[]},regression_validation:[]};
  write(join(store,core+'.retention.json'),{...retained,content_sha256:materialContentSHA(retained,true)});
  write(join(store,'active.json'),{id:expert,identity,core_version:core,width:64});
  write(join(store,'layer-23',expert+'.json'),{identity:identity+':layer-23',core_version:core,requires_gate:true,hidden:2048,width:64,sha256:sha});
  write(join(store,'active-router.json'),{id:router,identity,sha256:sha});
  const metadata={id:router,identity,core_version:core,expert_id:expert,expert_sha256:sha,sha256:sha,calibrated:true,gates_expert:true};
  const input={root,store,identity,modelPath,binary};
  for(const source of [undefined,'hf-bf16-prefix']){
   write(join(store,router+'.json'),{...metadata,feature_source:source});
   expect(()=>readMLXLearningSource(input)).toThrow('重新校准');
  }
  write(join(store,router+'.json'),{...metadata,feature_source:'mlx-q4-prefix'});
  write(join(modelPath,'download-receipt.json'),{complete:true,finished:Date.now()/1000+10,files:[{name:'model.safetensors',downloaded:true,bytes:1,sha256:sha}]});
  writeFileSync(join(modelPath,'model.safetensors'),'x');write(binary+'.json',{binary_sha256:sha});
  const binding={identity,core_version:core,core_file_sha256:sha,expert_id:expert,expert_file_sha256:sha,router_id:router,router_sha256:sha,mlx_model_sha256:sha,worker_binary_sha256:sha};
  write(join(root,'runtime/mlx/expert-proof.json'),{schema:'mlx-expert-local-trial-v1',...binding});
  write(join(root,'runtime/mlx/training-proof.json'),{schema:'mlx-native-training-qualification-v1',complete:true,...binding,gates:Object.fromEntries(MLX_TRAINING_GATES.map(key=>[key,true]))});
  expect(readMLXLearningSource(input).router.feature_source).toBe('mlx-q4-prefix');
 }finally{rmSync(root,{recursive:true,force:true});}
});

test('first-use learning notice applies only to an empty installation without any saved learning history',()=>{
 const root=mkdtempSync(join(tmpdir(),'mlx-fresh-readiness-')),store=join(root,'store');
 try{
  expect(isUntrainedMLXInstallation(store,()=>false)).toBe(true);
  expect(isUntrainedMLXInstallation(store,()=>true)).toBe(false);
  mkdirSync(join(store,'layer-23'),{recursive:true});
  expect(isUntrainedMLXInstallation(store,()=>false)).toBe(true);
  for(const name of ['active-core.json','active.json','active-router.json','layer-23/learned-old.safetensors','learning-old.json']){
   const path=join(store,name);writeFileSync(path,'broken historical data');
   expect(isUntrainedMLXInstallation(store,()=>false)).toBe(false);
   rmSync(path);
  }
  symlinkSync(join(root,'missing-pointer'),join(store,'active.json'));
  expect(isUntrainedMLXInstallation(store,()=>false)).toBe(false);
  expect(MLX_FIRST_LEARNING_NOTICE).toContain('缺少初始专家');
 }finally{rmSync(root,{recursive:true,force:true});}
});
