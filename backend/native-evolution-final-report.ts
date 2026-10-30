/** Recompute the current twelve gates from the complete durable FINAL. No source authority is minted here. */
import assert from "node:assert/strict";
import { validateCore450TrainingHistory } from "../.native-migration/research/native-core450-runner/training-safety.ts";
import { finishNativeEvolutionQuality, nativeEvolutionProgress, PARTIAL_GATES } from "./native-evolution-quality.ts";
import { uniqueNativeEvolutionRecords } from "./native-evolution-material.ts";
import { encoded } from "./canonical-json.ts";

export function validateCore450DurableFinal(saved: any, records: any, id: string, candidateSHA: string, referenceSHA: string) {
  const report = saved.report;
  assert(saved.step === 450 && report.completed_steps === 450 && report.actual_same_peer_observations === true);
  for (const key of ["quality_checked", "promoted", "product_admission", "adopted"])
    assert.equal(report[key], false, "FINAL transport cannot manufacture prior adoption");
  const safety = validateCore450TrainingHistory(saved, records.groups);
  assert(safety.complete_training_safety, "FINAL缺少完整450次训练安全记录");
  const binding = { core_version: "core-" + id, core_sha256: candidateSHA, expert_id: null, expert_sha256: null };
  const quality = finishNativeEvolutionQuality({ records, baseline: report.partial_baseline,
    first: report.partial_candidate_first, second: report.partial_candidate_second,
    baselineCoreGeneration: report.baseline_core_retention_generation,
    baselineCoreLoss: report.baseline_core_retention_loss, candidateCoreLoss: report.candidate_core_retention_loss,
    baselineReplayLoss: report.baseline_replay_loss, candidateReplayLoss: report.validation_replay_loss,
    completedSteps: safety.completed_steps, trainingSafetyFailed: safety.partial_safety_failed,
    identity: report.identity, binding, previousCoreSHA: referenceSHA, frozenRecordsSHA: report.partial_records_sha256 });
  assert.deepEqual(Object.keys(report.gates).sort(), [...PARTIAL_GATES].sort(), "FINAL不是当前完整12门");
  assert.equal(encoded(quality.gates), encoded(report.gates), "FINAL实际观察与12门不一致");
  assert.equal(quality.promoted, report.arithmetic_gates_passed);
  const progress = nativeEvolutionProgress(uniqueNativeEvolutionRecords(records.groups, records.protection, records.retention_regressions),
    report.partial_baseline, report.partial_candidate_first, report.partial_candidate_second, binding, false);
  assert.equal(encoded(progress), encoded(report.learning_progress), "FINAL未采纳进展与完整观察不一致");
  return { ...quality, training_safety: safety };
}
