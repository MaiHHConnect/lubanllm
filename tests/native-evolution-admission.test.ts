import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";

function child(code: string) {
  const result = spawnSync(process.execPath, ["--eval", code], { cwd: process.cwd(), encoding: "utf8", timeout: 30000 });
  expect(result.status, result.stderr).toBe(0);
}

test("graph authority is absent by default and rejects caller-created qualification objects", () => {
  child(`import assert from 'node:assert/strict';import{readNativeEvolutionGraphQualification,assertNativeEvolutionGraphQualification}from'./backend/native-evolution-graph-qualification.ts';
const binding={identity:'a'.repeat(64),graph_binary_sha256:'b'.repeat(64),training_binary_sha256:'c'.repeat(64),checkpoint_sha256:'d'.repeat(64),candidate_tensor_sha256:'e'.repeat(64),previous_tensor_sha256:'f'.repeat(64)};
assert.throws(()=>readNativeEvolutionGraphQualification(binding),/尚无唯一/);
assert.throws(()=>assertNativeEvolutionGraphQualification({...binding,independent_graph_qualified:true,product_admitted:false},binding),/不能传入/);`);
});

test("compiled report admission binds source and numerical evidence without accepting a passed flag", () => {
  child(`import assert from 'node:assert/strict';import{mock}from'bun:test';import{resolve}from'node:path';
const sha='a'.repeat(64),binding={identity:sha,graph_binary_sha256:sha,training_binary_sha256:sha,checkpoint_sha256:sha,candidate_tensor_sha256:sha,previous_tensor_sha256:sha};
const small={result_id:'small-original-result',actual_execution:'small-native',exit_code:0,cleanup_confirmed:true,pairs:[{name:'logits',elements:8,max_abs_error:0,allowed_max_abs_error:0}]};
const actual={...small,actual_execution:'original-HF-real-model'};
const report={protocol:'native-evolution-independent-graph-v1',...binding,release_manifest_sha256:sha,graph_source_proof_sha256:sha,training_source_proof_sha256:sha,inherited_math:{evidence_id:'issued-math-v1',graph_root_result_id:'original-hf-graph',training_root_result_id:'original-hf-adam',full_pairs:small.pairs,actual_original_HF:true,actual_full_graph_backwards:true,CE_KL_Adam_projection_steps:1,domains:['forward','CE','KL','gradient','Adam','projection'],unchanged_operator_sources:{CE:{previous:sha,current:sha}}},observation:{small,actual},tail:{small,actual}};
mock.module(resolve('backend/native-evolution-graph-reports.ts'),()=>({VERIFIED_NATIVE_EVOLUTION_GRAPH_REPORTS:[report]}));
mock.module(resolve('.native-migration/research/native-unified-service/compiled-authority-v4.ts'),()=>({RELEASE_AUTHORITY:{manifest_sha256:sha,sources:{persistentExpert:sha,trainer:sha}}}));
const{readNativeEvolutionGraphQualification:read,assertNativeEvolutionGraphQualification:check}=await import(resolve('backend/native-evolution-graph-qualification.ts'));
const cap=read(binding);check(cap,binding);
assert.throws(()=>check(cap,{...binding,identity:'b'.repeat(64)}));
report.observation.actual={...actual,pairs:[{name:'logits',elements:8,max_abs_error:1,allowed_max_abs_error:0}]};
assert.throws(()=>check(cap,binding));assert.throws(()=>read(binding),/对照/);
report.observation.actual={...actual,pairs:[{name:'head',elements:248320,max_abs_error:.0625,allclose_passed:true,atol:.01,rtol:.02,relative_l2:.00006,relative_l2_limit:.001}]};
read(binding);report.observation.actual.pairs[0].relative_l2=.002;assert.throws(()=>read(binding),/L2/);
report.observation.actual.pairs[0].relative_l2=.00006;report.observation.actual.pairs[0].allclose_passed=false;assert.throws(()=>read(binding));
report.observation.actual=actual;report.inherited_math.actual_original_HF=false;assert.throws(()=>read(binding));`);
});

test("candidate retention preserves original protection and prior read-only regressions across another round", () => {
  child(`import assert from 'node:assert/strict';import{mock}from'bun:test';import{resolve}from'node:path';
import{encoded}from'./backend/canonical-json.ts';import{provenance,messages,validateNativeCoreRetention}from'./backend/native-evolution-material.ts';
const old='.native-migration/',stub=(name,value)=>mock.module(resolve(name),()=>value);
stub('backend/native-evolution-admission.ts',{validateCore450QualityObservations:()=>{throw Error('not used')}});
stub(old+'backend/native-model-write.ts',{writeNativeModelTensors:()=>{throw Error('no model writes')}});
stub(old+'backend/native-model-tensor.ts',{readNativeModelTensors:()=>{throw Error('no model reads')}});
stub(old+'research/native-product-runtime/backend/evolution-snapshot-files.ts',{encoded,hash:v=>Bun.CryptoHasher.hash('sha256',v,'hex'),readSafe:()=>{},pathIn:()=>{},validateCoreRetention:()=>{}});
stub(old+'research/native-product-runtime/backend/native-retention-records.ts',{provenance,provenanceJSON:JSON.stringify});
stub(old+'research/native-full-training/io.ts',{durable:()=>{throw Error('no writes')}});
const{core450CandidateRetention}=await import(resolve('backend/native-evolution-artifacts.ts'));
const row=(id,purpose='validation',role='replay',readonly=true)=>({id,prompt:id,answer:'ok',source:'human',purpose,approved:true,training_allowed:!readonly,provenance:{evolution_role:role,...(readonly?{retention_generation_only:true}:{}),generation_check:{type:'exact',expected:'ok'}}});
const groups={new_training:[row('new-t','training','new',false)],new_validation:[row('new-v','validation','new',false)],replay_training:[row('old-t','training','replay',false)],replay_validation:[row('old-v','validation','replay',false)]};
const original=row('product-ordinary-0'),prior=row('prior-boundary'),fresh=row('new-boundary');
const records={groups,protection:[original,prior,groups.replay_validation[0]],retention_regressions:[fresh]};
const first=core450CandidateRetention(records,['new-t','new-v','new-boundary','old-t','old-v'],{identity:'a'.repeat(64)},'core-123456789abc').retention;
validateNativeCoreRetention(first,'a'.repeat(64),'core-123456789abc');
assert.deepEqual(new Set(first.regression_validation.map(r=>r.id)),new Set(['product-ordinary-0','prior-boundary','new-boundary','mastered-new-t','mastered-old-t']));
const second=core450CandidateRetention({groups:first.groups,protection:[...first.groups.new_validation,...first.groups.replay_validation,...first.regression_validation],retention_regressions:[]},['new-t','new-v','old-t','old-v'],{identity:'a'.repeat(64)},'core-abcdef123456').retention;
assert.deepEqual(second.regression_validation.map(r=>r.id),first.regression_validation.map(r=>r.id));
assert(second.regression_validation.every(r=>r.training_allowed===false));`);
});
