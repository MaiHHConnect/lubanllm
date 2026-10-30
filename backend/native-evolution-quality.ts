/** The current product's twelve quality gates, evaluated from actual observations. */
import assert from "node:assert/strict";
import { encoded } from "./canonical-json.ts";
import { ledgerParse, ledgerStringify } from "./memory.ts";
import { evaluateGeneration, isFiniteGenerationCheck, validateGenerationCheck } from "./generation-checks.ts";
import { messages, provenance, uniqueNativeEvolutionRecords, validateNativeProductRecords } from "./native-evolution-material.ts";
import { evaluateRetention, retentionAcceptsGrowth, retentionClass } from "./retention-policy.ts";

export const PARTIAL_POLICY = "closed-book-growth-v1";
export const PARTIAL_GATES = ["closed_book_growth", "original_correct_retained", "all_previous_mastery_retained",
  "two_complete_closed_book_proofs", "minimum_new_exposures", "training_safety_retained",
  "protected_replay_loss_retained", "core_tensor_changed", "material_binding_verified",
  "protected_core_retention_generation", "protected_core_retention_loss", "protected_replay_generation"] as const;
type Row = Record<string, any>;
type Observation = { id: string; answer: string; passed: boolean; complete: boolean; branch_enabled: boolean; reference_used: boolean };
type Binding = { core_version: string; core_sha256: string; expert_id: null; expert_sha256: null };
type Loss = { id: string; loss: number | Number };
export const evolutionContentSHA = (value: unknown) => Bun.CryptoHasher.hash("sha256", encoded(value), "hex");
const clone = <T>(value: T): T => ledgerParse(ledgerStringify(value));
const sorted = (ids: Iterable<string>) => [...ids].sort((a, b) => {
  const aa = Array.from(a, c => c.codePointAt(0)!), bb = Array.from(b, c => c.codePointAt(0)!);
  for (let i = 0; i < Math.min(aa.length, bb.length); i++) if (aa[i] !== bb[i]) return aa[i]! - bb[i]!;
  return aa.length - bb.length;
});
const intersection = (a: Set<string>, b: Set<string>) => new Set([...a].filter(id => b.has(id)));
const difference = (a: Set<string>, b: Set<string>) => new Set([...a].filter(id => !b.has(id)));
const finiteLoss = (value: unknown): number => {
  assert((typeof value === "number" || value instanceof Number) && Number.isFinite(Number(value)) && Number(value) >= 0,
    "核心保护损失记录不完整或含非有限值");
  return Number(value);
};

export function correctNativeEvolutionIds(observations: Observation[], records: Row[], finite = true) {
  assert(Array.isArray(observations), "部分进化缺少逐项观察");
  assert.deepEqual(observations.map(row => row.id), records.map(row => row.id), "部分进化缺少完整且有序的逐项观察");
  assert.equal(new Set(records.map(row => row.id)).size, records.length, "部分进化观察ID重复");
  const result = new Set<string>();
  for (let i = 0; i < records.length; i++) {
    const row = records[i]!, proof = observations[i]!;
    for (const key of ["passed", "complete", "branch_enabled", "reference_used"] as const)
      assert.equal(typeof proof[key], "boolean", "部分进化观察标志必须为真实布尔值");
    assert(!proof.branch_enabled && !proof.reference_used, "部分进化必须实际闭卷分支OFF");
    const check = validateGenerationCheck(provenance(row).generation_check, row.answer);
    const passed = evaluateGeneration(proof.answer, check, proof.complete).passed;
    assert.equal(proof.passed, passed, "部分进化passed与实际完整回答不一致");
    if (passed && (!finite || isFiniteGenerationCheck(check))) result.add(row.id);
  }
  return result;
}

export function nativeEvolutionProgress(records: Row[], baseline: Observation[], first: Observation[], second: Observation[],
  binding: Binding, accepted = false) {
  assert(/^core-[a-f0-9]{12}$/.test(binding.core_version) && /^[a-f0-9]{64}$/.test(binding.core_sha256)
    && binding.expert_id === null && binding.expert_sha256 === null, "部分进化候选绑定无效");
  const before = correctNativeEvolutionIds(baseline, records, false);
  const after = intersection(correctNativeEvolutionIds(first, records, false), correctNativeEvolutionIds(second, records, false));
  const mastered = intersection(correctNativeEvolutionIds(first, records), correctNativeEvolutionIds(second, records));
  const gained = difference(mastered, before), forgotten = difference(before, after), retained = intersection(before, after);
  const ids = records.map(row => row.id), pending = difference(new Set(ids), mastered);
  const proofs = records.flatMap((row, index) => mastered.has(row.id) ? [{ id: row.id,
    context_sha256: evolutionContentSHA(messages(row)), sample_sha256: evolutionContentSHA(row),
    check: validateGenerationCheck(provenance(row).generation_check, row.answer), binding: clone(binding),
    observations: [clone(first[index]!), clone(second[index]!)] }] : []);
  return { policy: PARTIAL_POLICY, attempted_ids: ids, baseline_correct_ids: sorted(before), mastered_ids: sorted(mastered),
    candidate_learned_ids: sorted(gained), learned_ids: accepted ? sorted(gained) : [], pending_ids: sorted(pending),
    retained_ids: sorted(retained), forgotten_ids: sorted(forgotten), attempted: ids.length,
    candidate_newly_mastered: gained.size, newly_mastered: accepted ? gained.size : 0, pending: pending.size,
    retained: retained.size, forgotten: forgotten.size, mastery_proofs: proofs,
    scope: "two complete closed-book observations of each exact full context on one frozen core; not transfer" };
}

export function nativeRetentionLossGuard(baseline: Loss[], candidate: Loss[]) {
  assert(Array.isArray(baseline) && Array.isArray(candidate), "核心保护损失记录必须是列表");
  const before = new Map<string, number>(), after = new Map<string, number>();
  for (const [rows, target] of [[baseline, before], [candidate, after]] as const) for (const row of rows) {
    assert(typeof row.id === "string" && row.id && !target.has(row.id), "核心保护损失ID无效或重复");
    target.set(row.id, finiteLoss(row.loss));
  }
  assert.deepEqual(sorted(before.keys()), sorted(after.keys()), "核心保护损失记录不完整");
  if (!before.size) return { passed: true, rows: [], scope: "没有已保护材料；没有能力保护证据" };
  const rows = [...before].map(([id, bound]) => {
    const value = after.get(id)!, tolerance = 64 * 2 ** -23 * Math.max(1, Math.abs(bound));
    const excess = value - bound - tolerance;
    return { id, baseline_loss: bound, candidate_loss: value, tolerance,
      excess_over_tolerance: excess, nonincreasing: excess <= 0 };
  });
  return { passed: rows.every(row => row.nonincreasing), rows,
    scope: "每条已保护材料；固定FP32测量上界，不是平均loss或能力全域证明" };
}

/** Call while the final candidate remains frozen under the worker's round lock.
 * This arithmetic does not establish source/graph admission or commit a model. */
export function finishNativeEvolutionQuality(input: {
  records: Row; baseline: Observation[]; first: Observation[]; second: Observation[];
  baselineCoreGeneration: Observation[]; baselineCoreLoss: Loss[]; candidateCoreLoss: Loss[];
  baselineReplayLoss: number | Number; candidateReplayLoss: number | Number;
  completedSteps: number; trainingSafetyFailed: boolean; identity: string; binding: Binding;
  previousCoreSHA: string; frozenRecordsSHA: string;
}) {
  const { records, baseline, first, second, binding } = input;
  validateNativeProductRecords(records);
  assert.equal(input.completedSteps, 450, "产品进化未完成固定450次实际更新");
  assert.equal(typeof input.trainingSafetyFailed, "boolean", "部分进化缺少训练安全状态");
  assert(/^[a-f0-9]{64}$/.test(input.identity) && /^[a-f0-9]{64}$/.test(input.previousCoreSHA), "部分进化原模型绑定无效");
  const all = uniqueNativeEvolutionRecords(records.groups, records.protection, records.retention_regressions);
  const value = nativeEvolutionProgress(all, baseline, first, second, binding);
  const retention = evaluateRetention(all, new Set(value.baseline_correct_ids), new Set([...value.mastered_ids, ...value.retained_ids]));
  const protectedIds = records.protection.map((row: Row) => row.id);
  const hardProtectedIds = records.protection.filter(row => retentionClass(row) === "hard").map(row => row.id);
  assert.deepEqual(sorted([...correctNativeEvolutionIds(input.baselineCoreGeneration, records.protection, false)].filter(id => hardProtectedIds.includes(id))),
    sorted(hardProtectedIds), "部分进化原核心硬保护基线未通过");
  const lossRecords = uniqueNativeEvolutionRecords({}, records.protection, records.retention_regressions);
  const lossIds = lossRecords.map((row: Row) => row.id);
  assert.deepEqual(input.baselineCoreLoss.map(row => row.id), lossIds, "部分进化逐条保护损失遗漏冻结材料");
  assert.deepEqual(input.candidateCoreLoss.map(row => row.id), lossIds, "部分进化候选保护损失遗漏冻结材料");
  const loss = nativeRetentionLossGuard(input.baselineCoreLoss, input.candidateCoreLoss);
  const before = new Set(value.baseline_correct_ids), firstMap = new Map(first.map(row => [row.id, row])),
    secondMap = new Map(second.map(row => [row.id, row])), beforeMap = new Map(baseline.map(row => [row.id, row]));
  const replay = records.groups.replay_validation.filter((row: Row) => before.has(row.id));
  const replayIds = replay.map((row: Row) => row.id);
  const take = (map: Map<string, Observation>, rows: Row[]) => rows.map(row => { const proof = map.get(row.id); assert(proof); return proof; });
  const retainedBoth = (rows: Row[]) => correctNativeEvolutionIds(take(firstMap, rows), rows, false).size === rows.length
    && correctNativeEvolutionIds(take(secondMap, rows), rows, false).size === rows.length;
  const required = new Set<string>([...protectedIds, ...all.filter(row => provenance(row).retention_from_version).map(row => row.id)]);
  const after = new Set([...value.mastered_ids, ...value.retained_ids]);
  const baselineReplay = finiteLoss(input.baselineReplayLoss), candidateReplay = finiteLoss(input.candidateReplayLoss);
  const materialSHA = evolutionContentSHA(all);
  const protectedBefore = new Set([...before].filter(id => protectedIds.includes(id))), protectedAfter = new Set([...after].filter(id => protectedIds.includes(id)));
  const protectedRetention = evaluateRetention(records.protection, protectedBefore, protectedAfter);
  const replayBefore = new Set(replayIds), replayAfter = new Set([...after].filter(id => replayBefore.has(id)));
  const replayRetention = evaluateRetention(replay, replayBefore, replayAfter);
  const gates = {
    closed_book_growth: value.candidate_learned_ids.length > 0,
    original_correct_retained: retentionAcceptsGrowth(retention, value.candidate_newly_mastered),
    all_previous_mastery_retained: [...required].every(id => after.has(id)),
    two_complete_closed_book_proofs: value.mastered_ids.length > 0,
    minimum_new_exposures: Math.floor(Math.floor(input.completedSteps / 2) / records.groups.new_training.length) >= 2,
    training_safety_retained: !input.trainingSafetyFailed,
    protected_replay_loss_retained: candidateReplay <= baselineReplay * 1.05 + .01,
    core_tensor_changed: binding.core_sha256 !== input.previousCoreSHA,
    material_binding_verified: materialSHA === input.frozenRecordsSHA,
    protected_core_retention_generation: protectedRetention.accepted,
    protected_core_retention_loss: loss.passed,
    protected_replay_generation: replayRetention.accepted,
  };
  const promoted = Object.values(gates).every(value => value === true);
  const evidence = {
    protected_replay_ids: replayIds,
    replay_generation_scope: "baseline-correct replay_validation only; both frozen candidate observations required",
    core_retention_generation: take(firstMap, records.protection),
    core_retention_generation_second: take(secondMap, records.protection),
    baseline_replay_generation: take(beforeMap, replay), replay_generation: take(firstMap, replay),
    replay_generation_second: take(secondMap, replay), core_retention_loss: loss,
  };
  return { identity: input.identity, version: binding.core_version, steps: 450, completed_steps: input.completedSteps,
    partial_policy: PARTIAL_POLICY, partial_safety_failed: input.trainingSafetyFailed,
    partial_records_sha256: materialSHA, partial_required_ids: sorted(required), retention_policy: retention,
    partial_baseline: clone(baseline), partial_candidate_first: clone(first), partial_candidate_second: clone(second),
    baseline_core_retention_generation: clone(input.baselineCoreGeneration),
    baseline_core_retention_loss: clone(input.baselineCoreLoss), candidate_core_retention_loss: clone(input.candidateCoreLoss),
    baseline_replay_loss: baselineReplay, validation_replay_loss: candidateReplay,
    split_ids: Object.fromEntries(Object.entries(records.groups).map(([key, rows]) => [key, (rows as Row[]).map(row => row.id)])),
    ...evidence, gates, promoted,
    learning_progress: nativeEvolutionProgress(all, baseline, first, second, binding, promoted),
    accepted_retention_groups: Object.fromEntries(Object.entries(records.groups).map(([key, rows]) =>
      [key, (rows as Row[]).filter(row => after.has(row.id)).map(clone)])),
  };
}
