/** Consume already issued, build-bound independent reports. This is not a mathematical test or an issuer. */
import assert from "node:assert/strict";
import { VERIFIED_NATIVE_EVOLUTION_GRAPH_REPORTS } from "./native-evolution-graph-reports.ts";
import { RELEASE_AUTHORITY } from "../.native-migration/research/native-unified-service/compiled-authority-v4.ts";
import { encoded } from "./canonical-json.ts";

type Pair = { name: string; elements: number; max_abs_error: number } & (
  { allowed_max_abs_error: number; allclose_passed?: never }
  | { allclose_passed: true; atol: number; rtol: number; relative_l2: number; relative_l2_limit: number; allowed_max_abs_error?: never }
);
type ActualComparison = { result_id: string; actual_execution: "small-native" | "original-HF-real-model";
  exit_code: 0; cleanup_confirmed: true; pairs: Pair[] };
export type NativeEvolutionConversionQualification = {
  method: "gptq-bf16-refine-v1"; evidence_id: string;
  training_binary_sha256: string; training_source_proof_sha256: string;
  shape: readonly [2048, 6144]; passes: 2; optimizer_steps_in_conversion: 0;
  calibration: "known-old-and-actual-q4-replay-training"; new_validation_used: false;
  independent_reference: { result_id: string; execution: "real-shape-independent-python-reference"; candidate_tensor_sha256: string; pairs: Pair[] };
  boundary: { result_id: string; step449_unmodified: true; step450_applied_once: true;
    seven_state_resume_equal: true; reference_training_basis_preserved: true; cancellation_before_commit: true; parent_rng_restored: true };
  actual_restore: { result_id: string; actual_execution: "native-real-model";
    graph_binary_sha256: string; candidate_tensor_sha256: string; restored_tensor_sha256: string;
    pid: number; engine_closed: true; lease_released: true; product_pointers_unchanged: true };

};
export function validateNativeEvolutionConversionQualification(value: NativeEvolutionConversionQualification, binding: {
  training_binary_sha256: string; training_source_proof_sha256: string; graph_binary_sha256: string;
}) {
  assert(value && value.method === "gptq-bf16-refine-v1" && evidenceID(value.evidence_id));
  assert.equal(value.training_binary_sha256, binding.training_binary_sha256);
  assert.equal(value.training_source_proof_sha256, binding.training_source_proof_sha256);
  assert(sha(value.training_binary_sha256) && sha(value.training_source_proof_sha256));
  assert.deepEqual(value.shape, [2048, 6144]); assert.equal(value.passes, 2);
  assert.equal(value.optimizer_steps_in_conversion, 0);
  assert.equal(value.calibration, "known-old-and-actual-q4-replay-training"); assert.equal(value.new_validation_used, false);
  assert(evidenceID(value.independent_reference.result_id) && value.independent_reference.execution === "real-shape-independent-python-reference");
  checkPairs(value.independent_reference.pairs);
  const names=new Set(value.independent_reference.pairs.map(pair=>pair.name));
  for(const name of ["converted-candidate", "converted-master"]){
    assert(names.has(name), "缺少转换独立数值对照");
    const pair=value.independent_reference.pairs.find(pair=>pair.name===name)!;
    assert.equal(pair.elements,2048*6144);assert("allowed_max_abs_error"in pair && pair.allowed_max_abs_error===0 && pair.max_abs_error===0,"转换必须逐元素精确复现独立参考");
  }
  assert(evidenceID(value.boundary.result_id));
  for(const key of ["step449_unmodified", "step450_applied_once", "seven_state_resume_equal", "reference_training_basis_preserved", "cancellation_before_commit", "parent_rng_restored"] as const)assert.equal(value.boundary[key],true);
  const restored=value.actual_restore;
  assert(restored.actual_execution === "native-real-model" && evidenceID(restored.result_id));
  assert.equal(restored.graph_binary_sha256,binding.graph_binary_sha256);
  assert(sha(restored.graph_binary_sha256)&&sha(restored.candidate_tensor_sha256)&&sha(restored.restored_tensor_sha256));
  assert.equal(restored.candidate_tensor_sha256,value.independent_reference.candidate_tensor_sha256);
  assert(Number.isSafeInteger(restored.pid)&&restored.pid>0&&restored.engine_closed===true&&restored.lease_released===true&&restored.product_pointers_unchanged===true);

}
export type NativeEvolutionGraphReport = {
  protocol: "native-evolution-independent-graph-v1" | "native-evolution-independent-graph-v2"; identity: string;
  terminal_conversion?: NativeEvolutionConversionQualification;
  release_manifest_sha256: string; graph_binary_sha256: string; training_binary_sha256: string;
  graph_source_proof_sha256: string; training_source_proof_sha256: string;
  inherited_math: { evidence_id: string; graph_root_result_id: string; training_root_result_id: string; full_pairs: Pair[];
    actual_original_HF: true; actual_full_graph_backwards: true; CE_KL_Adam_projection_steps: 1;
    domains: readonly ["forward", "CE", "KL", "gradient", "Adam", "projection"];
    unchanged_operator_sources: Record<string, { previous: string; current: string }> };
  observation: { small: ActualComparison; actual: ActualComparison };
  tail: { small: ActualComparison; actual: ActualComparison };
};
export type NativeEvolutionGraphBinding = { identity: string; graph_binary_sha256: string; training_binary_sha256: string;
  checkpoint_sha256: string; candidate_tensor_sha256: string; previous_tensor_sha256: string };
export type NativeEvolutionGraphQualification = Readonly<{ identity: string; graph_binary_sha256: string;
  training_binary_sha256: string; independent_graph_qualified: true; product_admitted: false }>;
const issued = new WeakMap<NativeEvolutionGraphQualification, { report: NativeEvolutionGraphReport; encoded: string }>();
const sha = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const evidenceID = (value: unknown) => typeof value === "string" && !!value.trim() && !value.includes("\0");
function checkPairs(pairs: Pair[]) {
  assert(Array.isArray(pairs) && pairs.length > 0 && new Set(pairs.map(p => p.name)).size === pairs.length);
  for (const pair of pairs) {
    assert(pair.name && Number.isSafeInteger(pair.elements) && pair.elements > 0
      && Number.isFinite(pair.max_abs_error) && pair.max_abs_error >= 0, "独立数值对照无效");
    if (pair.allclose_passed === true) {
      assert(pair.allowed_max_abs_error === undefined
        && [pair.atol, pair.rtol, pair.relative_l2, pair.relative_l2_limit].every(v => Number.isFinite(v) && v >= 0)
        && pair.relative_l2 <= pair.relative_l2_limit, "独立allclose及相对L2对照未通过");
    } else assert(Number.isFinite(pair.allowed_max_abs_error) && pair.allowed_max_abs_error! >= 0
      && pair.max_abs_error <= pair.allowed_max_abs_error!, "独立绝对误差对照未通过");
  }
}
function checkReport(report: NativeEvolutionGraphReport) {
  if(report.protocol === "native-evolution-independent-graph-v1") {
    // The only already issued pre-conversion trainer. New trainer sources require v2 evidence.
    assert.equal(report.training_binary_sha256, "dea39ec08e4deeb2378a66c7460cafcfdb6a364b1332e66438fc53f14acb72ae");
    assert.equal(report.terminal_conversion, undefined);
  } else {
    assert.equal(report.protocol, "native-evolution-independent-graph-v2");
    assert(report.terminal_conversion, "末尾转换缺少独立实际形状与恢复资格");
    validateNativeEvolutionConversionQualification(report.terminal_conversion, report);
  }
  for (const field of ["identity", "graph_binary_sha256", "training_binary_sha256"] as const) assert(sha(report[field]));
  assert.equal(report.release_manifest_sha256, RELEASE_AUTHORITY.manifest_sha256);
  assert.equal(report.graph_source_proof_sha256, RELEASE_AUTHORITY.sources.persistentExpert);
  assert.equal(report.training_source_proof_sha256, RELEASE_AUTHORITY.sources.trainer);
  const math = report.inherited_math;
  assert(math.actual_original_HF === true && math.actual_full_graph_backwards === true && math.CE_KL_Adam_projection_steps === 1);
  assert.deepEqual(math.domains, ["forward", "CE", "KL", "gradient", "Adam", "projection"]);
  for (const key of ["evidence_id", "graph_root_result_id", "training_root_result_id"] as const) assert(evidenceID(math[key]));
  checkPairs(math.full_pairs);
  assert(Object.keys(math.unchanged_operator_sources).length > 0, "没有原数学证据对应的未改运算来源");
  for (const pair of Object.values(math.unchanged_operator_sources)) assert(sha(pair.previous) && pair.previous === pair.current);
  for (const family of [report.observation, report.tail]) for (const scale of ["small", "actual"] as const) {
    const result = family[scale];
    assert(evidenceID(result.result_id) && result.exit_code === 0 && result.cleanup_confirmed === true);
    assert.equal(result.actual_execution, scale === "small" ? "small-native" : "original-HF-real-model");
    checkPairs(result.pairs);
  }
}

/** Only the package's independently issued table is eligible; runtime JSON cannot manufacture a cap. */
export function readNativeEvolutionGraphQualification(binding: Pick<NativeEvolutionGraphBinding,
  "identity" | "graph_binary_sha256" | "training_binary_sha256">): NativeEvolutionGraphQualification {
  const reports = VERIFIED_NATIVE_EVOLUTION_GRAPH_REPORTS.filter(report => report.identity === binding.identity
    && report.graph_binary_sha256 === binding.graph_binary_sha256 && report.training_binary_sha256 === binding.training_binary_sha256);
  assert.equal(reports.length, 1, "当前原生来源尚无唯一实际图资格报告，不能采纳");
  const report = reports[0]!; checkReport(report);
  const cap = Object.freeze({ ...binding, independent_graph_qualified: true as const, product_admitted: false as const });
  issued.set(cap, { report, encoded: encoded(report) }); return cap;
}

export function assertNativeEvolutionGraphQualification(cap: NativeEvolutionGraphQualification, binding: NativeEvolutionGraphBinding) {
  const state = issued.get(cap); assert(state, "图资格必须来自包内实际独立报告，不能传入passed声明");
  assert.equal(encoded(state.report), state.encoded);
  for (const key of ["identity", "graph_binary_sha256", "training_binary_sha256"] as const) assert.equal(cap[key], binding[key]);
  for (const key of ["checkpoint_sha256", "candidate_tensor_sha256", "previous_tensor_sha256"] as const) assert(sha(binding[key]));
}
