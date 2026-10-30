/** Publish only mastery observed in ordinary Q4; retain the independent BF16 training evidence verbatim. */
import assert from "node:assert/strict";
import { finishNativeEvolutionQuality } from "./native-evolution-quality.ts";
import type { NativeEvolutionQ4Evidence } from "./native-evolution-admission.ts";

export class NativeEvolutionQualityRejection extends Error {
  constructor(message: string, readonly report: Record<string, any>) { super(message); this.name = "NativeEvolutionQualityRejection"; }
}

export function finishNativeQ4Quality(records: Record<string, any>, training: Record<string, any>, q4: NativeEvolutionQ4Evidence,
  binding: { core_version: string; core_sha256: string; expert_id: null; expert_sha256: null }, previousTensorSHA: string) {
  // A capability seen in either actual baseline is already known. Each selected row remains an unmodified actual observation.
  const baseline = q4.baseline[0].rows.map((row, index) => row.passed ? row : q4.baseline[1].rows[index]!);
  const quality = finishNativeEvolutionQuality({ records, baseline, first: [...q4.candidate[0].rows], second: [...q4.candidate[1].rows],
    baselineCoreGeneration: training.baseline_core_retention_generation,
    baselineCoreLoss: training.baseline_core_retention_loss, candidateCoreLoss: training.candidate_core_retention_loss,
    baselineReplayLoss: training.baseline_replay_loss, candidateReplayLoss: training.validation_replay_loss,
    completedSteps: training.completed_steps, trainingSafetyFailed: training.partial_safety_failed,
    identity: training.identity, binding, previousCoreSHA: previousTensorSHA, frozenRecordsSHA: training.partial_records_sha256 });
  assert(Object.values(training.gates).every(value => value === true), "BF16训练12门未通过");
  const report = { ...training, ...quality, training_precision_evidence: training,
    generation_precision: "mixed-q4-g16", training_precision: "torch.bfloat16",
    protection_loss_precision: "torch.bfloat16", original_protection_baseline_precision: "torch.bfloat16",
    q4_baseline_first: q4.baseline[0].rows, q4_baseline_second: q4.baseline[1].rows,
    q4_baseline_policy: "each complete actual correct baseline is protected; choose first correct observation per record" };
  if (!quality.promoted) throw new NativeEvolutionQualityRejection("实际普通Q4未产生新增掌握或未保住原有能力", report);
  return report;
}
