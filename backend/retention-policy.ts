/** Bounded retention policy shared by specialist learning and native evolution. */
import { materialProvenance } from "./native-evolution-material.ts";

export type RetentionClass = "hard" | "soft";

/** Identity facts and adopted/original core material are never allowed to regress. */
export function retentionClass(row: Record<string, any>): RetentionClass {
  const p = materialProvenance(row);
  if (p.original_core_baseline === true || p.retention_scope === "original_core"
      || p.protocol_id === "explicit-model-identity-v1"
      || p.teaching_fact === "model_name" || p.teaching_fact === "model_fine_tuner"
      || p.teaching_fact === "model_positioning" || typeof p.retention_from_version === "string") return "hard";
  return "soft";
}

/** Five or more old correct soft items receive a conservative ten percent budget. */
export function softForgetBudget(baselineSoftCount: number): number {
  if (!Number.isSafeInteger(baselineSoftCount) || baselineSoftCount < 0) throw new Error("软保护数量无效");
  return baselineSoftCount < 5 ? 0 : Math.max(1, Math.floor(baselineSoftCount * 0.1));
}

export function evaluateRetention(records: Record<string, any>[], baselineCorrect: Set<string>, candidateCorrect: Set<string>) {
  const hard = new Set(records.filter(row => retentionClass(row) === "hard").map(row => row.id));
  const soft = new Set(records.filter(row => retentionClass(row) === "soft").map(row => row.id));
  const forgotten = new Set([...baselineCorrect].filter(id => !candidateCorrect.has(id)));
  const hardForgotten = new Set([...forgotten].filter(id => hard.has(id)));
  const softForgotten = new Set([...forgotten].filter(id => soft.has(id)));
  const baselineSoft = new Set([...baselineCorrect].filter(id => soft.has(id)));
  const budget = softForgetBudget(baselineSoft.size);
  return {
    hard_ids: [...hard].sort(), soft_ids: [...soft].sort(),
    baseline_soft_count: baselineSoft.size, soft_forget_budget: budget,
    forgotten_ids: [...forgotten].sort(), hard_forgotten_ids: [...hardForgotten].sort(),
    soft_forgotten_ids: [...softForgotten].sort(),
    accepted: hardForgotten.size === 0 && softForgotten.size <= budget,
  };
}

export function retentionAcceptsGrowth(value: ReturnType<typeof evaluateRetention>, gainedCount: number) {
  if (!Number.isSafeInteger(gainedCount) || gainedCount < 0) throw new Error("新增掌握数量无效");
  return value.accepted && (value.soft_forgotten_ids.length === 0 || gainedCount > value.soft_forgotten_ids.length);
}
