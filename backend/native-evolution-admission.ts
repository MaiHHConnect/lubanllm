/** Live FINAL ownership, twelve quality gates, independent graph authority and actual ordinary-Q4 protection. */
import assert from "node:assert/strict";
import { NativeEngine, assertNativeExpertScopeGuard } from "../.native-migration/research/native-unified-expert/native-engine.ts";
import { borrowNativeEnginePeer } from "../.native-migration/research/native-unified-peer/engine-peer.ts";
import { validateCore450Checkpoint, type Core450Checkpoint } from "../.native-migration/research/native-core450-runner/checkpoint.ts";
import { core450ControlMetadata } from "../.native-migration/research/native-core450-owner-controller/metadata.ts";
import type { SnapshotGuard } from "../.native-migration/research/native-product-runtime/backend/evolution-snapshots.ts";
import type { SnapshotBase } from "../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts";
import type { Memory } from "../.native-migration/research/native-product-runtime/backend/memory.ts";
import { encoded } from "./canonical-json.ts";
import { ledgerParse } from "./memory.ts";
import { selectIndependentNativeEvolutionCohort } from "./native-evolution-cohort.ts";
import { uniqueNativeEvolutionRecords, provenance } from "./native-evolution-material.ts";
import { correctNativeEvolutionIds, nativeEvolutionProgress } from "./native-evolution-quality.ts";
import { validateCore450DurableFinal } from "./native-evolution-final-report.ts";
import { assertNativeQ4Observation, type NativeQ4Observation } from "./native-evolution-q4.ts";
import { assertNativeEvolutionGraphQualification, type NativeEvolutionGraphQualification } from "./native-evolution-graph-qualification.ts";
import { finishNativeQ4Quality, NativeEvolutionQualityRejection } from "./native-evolution-q4-quality.ts";

export type NativeEvolutionQ4Evidence = { baseline: readonly [NativeQ4Observation, NativeQ4Observation];
  candidate: readonly [NativeQ4Observation, NativeQ4Observation] };
export type Core450QualityObservations = {
  readonly checkpoint: Core450Checkpoint; readonly report: Readonly<Record<string, any>>;
  readonly evidence: Readonly<ReturnType<typeof validateCore450DurableFinal>>;
  readonly production_model_qualified: boolean; readonly product_admitted: false; assertUnchanged(): Promise<void>;
};
type Context = { memory: Memory; store: string; base: SnapshotBase };
const issued = new WeakMap<Core450QualityObservations, { engine: NativeEngine; guard: SnapshotGuard;
  peer: ReturnType<typeof borrowNativeEnginePeer>; context: Context;
  cohort: ReturnType<typeof selectIndependentNativeEvolutionCohort>; json: string;
  authority?: { graph: NativeEvolutionGraphQualification; q4: NativeEvolutionQ4Evidence; fileSHA: string;
    published: ReturnType<typeof finishNativeQ4Quality> } }>();

export function validateCore450QualityObservations(engine: NativeEngine, guard: SnapshotGuard, cap: Core450QualityObservations) {
  const state = issued.get(cap);
  assert(state && state.engine === engine && state.guard === guard, "质量检查需要本轮真实FINAL和独占模型作用域");
  assertNativeExpertScopeGuard(engine, guard); validateCore450Checkpoint(engine, cap.checkpoint);
  assert(borrowNativeEnginePeer(engine, guard) === state.peer);
  assert.equal(encoded({ report: cap.report, evidence: cap.evidence }), state.json);
  state.cohort.assertUnchanged(); return state;
}

export async function observeCore450ProductQuality(engine: NativeEngine, guard: SnapshotGuard,
  checkpoint: Core450Checkpoint, context: Context) {
  assertNativeExpertScopeGuard(engine, guard); validateCore450Checkpoint(engine, checkpoint);
  assert(checkpoint.approval.step === 450 && checkpoint.binding.scope === "product"
    && checkpoint.binding.device === "mps" && checkpoint.binding.dtype === "torch.bfloat16");
  const peer = borrowNativeEnginePeer(engine, guard);
  assert(engine.config.routing?.store === context.store && encoded(peer.base) === encoded(context.base)
    && context.memory.model_identity === context.base.identity && !peer.worker.hasInstalledCore,
  "采纳检查必须在原训练finally已恢复之后开始");
  const cohort = selectIndependentNativeEvolutionCohort(context.memory, context.store, context.base);
  assert.equal(cohort.records_sha256, checkpoint.binding.records_sha256);
  const saved = core450ControlMetadata(checkpoint.checkpoint.payload), report = ledgerParse(encoded(saved.report));
  const evidence = validateCore450DurableFinal(saved, cohort.records, checkpoint.approval.id,
    checkpoint.state.candidate.sha256, checkpoint.learning_reference.sha256);
  const cap: Core450QualityObservations = Object.freeze({ checkpoint, report: Object.freeze(report), evidence: Object.freeze(evidence),
    get production_model_qualified() { return !!issued.get(cap)?.authority; }, product_admitted: false as const,
    async assertUnchanged() { validateCore450QualityObservations(engine, guard, cap); } });
  issued.set(cap, { engine, guard, peer, context, cohort, json: encoded({ report, evidence }) });
  return cap;
}

/** Called only with private receipts after both Q4 worker owners have closed. */
export function authorizeCore450Publication(engine: NativeEngine, guard: SnapshotGuard, cap: Core450QualityObservations,
  fileSHA: string, graph: NativeEvolutionGraphQualification, q4: NativeEvolutionQ4Evidence) {
  const state = validateCore450QualityObservations(engine, guard, cap), cp = cap.checkpoint, { base } = state.context;
  assert(Object.values(cap.evidence.gates).every(value => value === true), "当前12门未全部通过");
  assert(/^[a-f0-9]{64}$/.test(fileSHA));
  assertNativeEvolutionGraphQualification(graph, { identity: base.identity,
    graph_binary_sha256: state.peer.build.binary_sha256,
    training_binary_sha256: cap.report.training_engine.binary_sha256,
    checkpoint_sha256: cp.approval.checkpoint_sha256,
    candidate_tensor_sha256: cp.state.candidate.sha256, previous_tensor_sha256: cp.learning_reference.sha256 });
  const records = uniqueNativeEvolutionRecords(state.cohort.records.groups, state.cohort.records.protection,
    state.cohort.records.retention_regressions);
  assert(q4?.baseline.length === 2 && q4.candidate.length === 2, "Q4必须有两次完整基线和候选观察");
  const sets = [...q4.baseline, ...q4.candidate].map((proof, index) => {
    assertNativeQ4Observation(proof, records);
    assert(proof.identity === base.identity && proof.manifest_sha256 === base.model_manifest_sha256);
    const expected = index < 2 ? state.cohort.core?.sha256 ?? null : fileSHA;
    assert.equal(proof.core?.sha256 ?? null, expected, "Q4实际权重文件不是本轮原模型或候选");
    if (proof.core) assert.equal(proof.core.tensor_key, base.core_parameter);
    assert.equal(encoded(proof.rows.map(row => row.input_tokens)), encoded(q4.baseline[0].rows.map(row => row.input_tokens)),
      "Q4模型使用了不同完整上下文或分词模板");
    return correctNativeEvolutionIds([...proof.rows], records, false);
  });
  const protectedIDs = new Set<string>([...sets[0]!, ...sets[1]!]);
  for (const id of [...cap.evidence.learning_progress.candidate_learned_ids,
    ...records.filter(row => provenance(row).retention_from_version).map(row => row.id),
    ...state.cohort.records.protection.map(row => row.id), ...state.cohort.records.retention_regressions.map(row => row.id)]) protectedIDs.add(id);
  const published = finishNativeQ4Quality(state.cohort.records, cap.report, q4, {
    core_version: "core-" + cp.approval.id, core_sha256: fileSHA, expert_id: null, expert_sha256: null }, cp.learning_reference.sha256);
  const missing = [...protectedIDs].filter(id => !sets[2]!.has(id) || !sets[3]!.has(id));
  if (missing.length) throw new NativeEvolutionQualityRejection("最终普通Q4遗漏了本轮新增或已有保护材料", {
    ...published, promoted: false, product_admission: false, adopted: false, q4_missing_required_ids: missing,
    learning_progress: nativeEvolutionProgress(records, published.partial_baseline, published.partial_candidate_first,
      published.partial_candidate_second, { core_version: "core-" + cp.approval.id, core_sha256: fileSHA,
        expert_id: null, expert_sha256: null }, false) });
  state.authority = { graph, q4, fileSHA, published };
}

export function requireCore450Publication(engine: NativeEngine, guard: SnapshotGuard, cap: Core450QualityObservations, fileSHA: string) {
  const state = validateCore450QualityObservations(engine, guard, cap);
  assert(state.authority && cap.production_model_qualified, "缺少当前图资格或实际Q4保护证据，禁止采纳");
  assert.equal(state.authority.fileSHA, fileSHA, "实际Q4观察与即将采纳的权重文件不同");
  assert(Object.values(cap.evidence.gates).every(value => value === true));
  return state.authority.published;
}
