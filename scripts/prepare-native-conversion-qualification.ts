/** Offline issuer for the existing actual conversion/reference + source-swap evidence. No model execution. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {VERIFIED_NATIVE_EVOLUTION_GRAPH_REPORTS} from '../backend/native-evolution-graph-reports.ts';
import {validateNativeEvolutionConversionQualification,type NativeEvolutionConversionQualification,type NativeEvolutionGraphReport} from '../backend/native-evolution-graph-qualification.ts';
const root=resolve(import.meta.dir,'..');
const read=async(path:string)=>JSON.parse(await readFile(path,'utf8'));
export async function prepareNativeConversionQualification(evidenceRoot:string){
 const [runtime,build,cpu,independent,actual]=await Promise.all([
  read(join(root,'build/native-evolution/runtime-authority.json')),read(join(root,'build/native-evolution-conversion/build.json')),
  read(join(root,'build/native-evolution-conversion/cpu-proof.json')),read(join(evidenceRoot,'terminal-conversion-independent-v1/result.json')),
  read(join(evidenceRoot,'gptq-bf16-refine-retention-v1/result.json'))]);
 const manifest=await read(join(runtime.runtimeRoot,runtime.authority.manifest_relative)),proof=manifest.programs.trainer.proof;
 assert(proof.terminal_conversion?.method==='gptq-bf16-refine-v1','Install the tested conversion trainer role before issuing its final binding');
 assert.equal(proof.original_binary_sha256,build.binary_sha256);assert.equal(cpu.binary_sha256,build.binary_sha256);
 assert.equal(proof.build.sources.training.sha256,build.sources.training.sha256);
 assert.equal(proof.build.sources.conversion.sha256,build.sources.conversion.sha256);
 assert.equal(proof.build.sources.basis.sha256,build.sources.basis.sha256);
 assert(build.inverse_exact&&cpu.unchanged_math.actual_steps_each_arm===2&&cpu.unchanged_math.deterministic_raw_equal===37&&cpu.unchanged_math.all_seven_state_bit_exact);
 assert(cpu.terminal_boundary&&cpu.resume_bit_exact&&cpu.cancel_before_commit&&cpu.parent_rng_restored);
 assert(independent.schema==='native-terminal-conversion-independent-reference-v1'&&independent.passed&&independent.inputs_stat_unchanged&&independent.previous_diagnostic_candidate_bit_exact);
 assert.deepEqual(independent.actual_shape,[2048,6144]);assert(independent.GPU===false&&independent.optimizer_steps===0&&independent.new_validation_used===false);
 assert.equal(resolve(independent.command[0]),resolve(build.cpu));
 assert(actual.passed&&actual.engine_closed&&actual.lease_released&&actual.product_pointers_unchanged&&actual.optimizer_steps===0&&!actual.product_admitted);
 const previous=VERIFIED_NATIVE_EVOLUTION_GRAPH_REPORTS.find(r=>r.protocol==='native-evolution-independent-graph-v1'&&r.identity===actual.identity);
 assert(previous,'Missing the originally issued unchanged-operator evidence');
 assert.equal(runtime.persistentBinarySHA,previous.graph_binary_sha256,'Changed graph needs a new actual source-swap/restore test');
 assert.equal(runtime.authority.sources.persistentExpert,previous.graph_source_proof_sha256,'Changed graph sources cannot inherit the old actual restore');
 assert.equal(build.original_sources.training.sha256,previous.inherited_math.unchanged_operator_sources['Training.v3.cpp']!.current);
 const peer=join(actual.work,'engine-peer');
 const [startup,applied,confirmed,restored,restoreConfirmed]=await Promise.all(['startup.json','command-32.json','command-33.json','command-76.json','command-77.json'].map(p=>read(join(peer,p))));
 assert.equal(startup.binary_sha256,runtime.persistentBinarySHA);assert.equal(startup.ready.device,'mps');
 const candidate=actual.projection.core.sha256,original=actual.previous_sha256,pid=startup.ready.pid;
 const check=(row:any,operation:string,sha:string,apply:boolean)=>{
  assert.equal(row.pid,pid);assert.equal(row.request,operation);assert.equal(row.response.operation,operation);
  assert.equal(row.response.event,'done');assert.equal(row.response.core_tensor_sha256,sha);
  assert(row.response.backbone_resident&&row.response.branch_forwards===0&&row.response.optimizer_step===0);
  if(apply){const ack=row.response.core_source_ack;assert.equal(ack.protocol,'luban-retained-core-source-v2');
   assert.equal(ack.pid,pid);assert.equal(ack.base_identity,actual.identity);assert.equal(ack.actual_core_tensor_sha256,sha);
   assert.deepEqual(ack.actual_core_shape,[2048,6144]);assert.equal(ack.actual_core_bytes,25165824);
   assert(ack.same_official_storage&&!ack.resident_core_readback&&!ack.parent_rng_snapshot&&!ack.official_graph_reloaded);
  }else assert.equal(row.response.core_source_confirming,false);
 };
 check(applied,'core-source-apply',candidate,true);check(confirmed,'core-source-confirm',candidate,false);
 check(restored,'core-source-apply',original,true);check(restoreConfirmed,'core-source-confirm',original,false);
 assert.equal(actual.restored.pid,pid);assert.equal(actual.restored.core_sha256,original);
 assert.equal(actual.projection.metrics.schema,'known-old-q4-gptq-bf16-refine-v1');
 const binding={training_binary_sha256:runtime.roles.trainer.sha256,training_source_proof_sha256:runtime.authority.sources.trainer,graph_binary_sha256:runtime.persistentBinarySHA};
 const conversion:NativeEvolutionConversionQualification={method:'gptq-bf16-refine-v1',evidence_id:'terminal-conversion-independent-v1+native-refine-restore-v1',...binding,
  shape:[2048,6144],passes:2,optimizer_steps_in_conversion:0,calibration:'known-old-and-actual-q4-replay-training',new_validation_used:false,
  independent_reference:{result_id:join(evidenceRoot,'terminal-conversion-independent-v1/result.json'),execution:'real-shape-independent-python-reference',candidate_tensor_sha256:candidate,pairs:independent.pairs},
  boundary:{result_id:'build/native-evolution-conversion/cpu-proof.json',step449_unmodified:true,step450_applied_once:true,seven_state_resume_equal:true,reference_training_basis_preserved:true,cancellation_before_commit:true,parent_rng_restored:true},
  actual_restore:{result_id:join(evidenceRoot,'gptq-bf16-refine-retention-v1/result.json'),actual_execution:'native-real-model',graph_binary_sha256:runtime.persistentBinarySHA,candidate_tensor_sha256:candidate,restored_tensor_sha256:original,pid,engine_closed:true,lease_released:true,product_pointers_unchanged:true}};
 validateNativeEvolutionConversionQualification(conversion,binding);
 const report:NativeEvolutionGraphReport={...structuredClone(previous),protocol:'native-evolution-independent-graph-v2',...binding,
  release_manifest_sha256:runtime.authority.manifest_sha256,terminal_conversion:conversion};
 // This is the unchanged frozen operator portion recovered by the reversible overlay, not the new whole translation unit.
 const frozen=report.inherited_math.unchanged_operator_sources['Training.v3.cpp']!;
 delete report.inherited_math.unchanged_operator_sources['Training.v3.cpp'];
 report.inherited_math.unchanged_operator_sources['Training.v3.cpp/original-operator-body-before-terminal-overlay']=frozen;
 return report;
}
if(import.meta.main){
 const args=process.argv.slice(2),issue=args.includes('--issue'),values=args.filter(v=>v!=='--issue');
 assert.equal(values.length,2,'Usage: bun scripts/prepare-native-conversion-qualification.ts V19_EVIDENCE_ROOT NEW_REPORT.json [--issue]');
 const report=await prepareNativeConversionQualification(resolve(values[0]!));
 await writeFile(resolve(values[1]!),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
 if(issue)await writeFile(join(root,'backend/native-evolution-graph-reports.ts'),`/** Issued independent evidence; runtime data cannot add qualifications. */\nimport type {NativeEvolutionGraphReport} from './native-evolution-graph-qualification.ts';\nexport const VERIFIED_NATIVE_EVOLUTION_GRAPH_REPORTS:readonly NativeEvolutionGraphReport[]=Object.freeze(${JSON.stringify([report],null,2)});\n`);
 console.log(JSON.stringify({report:resolve(values[1]!),issued:issue,new450_passed:false,product_admitted:false}));
}
